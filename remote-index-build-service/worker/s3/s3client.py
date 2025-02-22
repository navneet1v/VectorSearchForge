import traceback
from concurrent.futures.thread import ThreadPoolExecutor

import boto3
import os
import tempfile
from pathlib import Path
from botocore.exceptions import ClientError
import math

import logging

# make this region dynamic later
s3_client = boto3.client('s3', region_name="us-west-2")
logger = logging.getLogger(__name__)

def check_s3_object_exists(bucket_name, object_key):
    """
    Check if an object exists in an S3 bucket.

    This function performs a HEAD request on the S3 object to verify its existence
    without downloading the object content. It handles the AWS ClientError exception
    to determine if the object exists.

    Args:
        bucket_name (str): The name of the S3 bucket to check.
        object_key (str): The key (path) of the object within the bucket.

    Returns:
        bool: True if the object exists, False if it doesn't.

    Raises:
        botocore.exceptions.ClientError: If there's an error other than 404 (like permissions issues,
            invalid bucket name, network problems, etc.)
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

def download_s3_file_in_chunks(bucket_name, object_key, chunk_size=1024*1024*10):  # 10MB chunks
    """
    Download a file from S3 in chunks and save to temp directory.
    TODO: This is downloading the file in sequence. We will make this in parallel in next iteration
    
    Args:
        bucket_name (str): The S3 bucket name
        object_key (str): The S3 object key (file path)
        chunk_size (int): Size of chunks to download (default 1MB)
        
    Returns:
        str: Path to the downloaded file in temp directory
    """
    
    try:
        # Get object details
        logger.info(f"Bucket name: {bucket_name}, Object key: {object_key}")
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']
        
        # Create temp file with same extension as original
        file_extension = Path(object_key).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        
        # Get the object
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        downloaded = 0
        with open(temp_file_path, 'wb') as f:
            logger.info(f"Downloading {object_key} to {temp_file_path}")
            
            # Download and write chunks
            # This needs to be parallelized
            for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Calculate and display progress
                progress = (downloaded / file_size) * 100
                logger.info(f"Progress: {progress:.2f}% ({downloaded}/{file_size} bytes)")

        logger.info(f"Download completed: {temp_file_path}")
        return temp_file_path
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.info(f"The object {object_key} does not exist in bucket {bucket_name}")
        elif error_code == 'NoSuchBucket':
            logger.info(f"The bucket {bucket_name} does not exist")
        else:
            logger.info(f"Error downloading object: {traceback.format_exc()} {e}")
            
        # Clean up temp file if it exists
        cleanup_temp_file(temp_file_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()} {e}")
        # Clean up temp file if it exists
        cleanup_temp_file(temp_file_path)
        raise

def cleanup_temp_file(temp_file_path):
    """
    Clean up the temporary file when no longer needed
    
    Args:
        temp_file_path (str): Path to the temporary file
    """
    try:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Temporary file removed: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error removing temporary file: {traceback.format_exc()} {e}")



"""
max_workers: Maximum number of concurrent upload threads
chunk_size: Size of each chunk in bytes (default 1GB)
"""
max_workers = os.cpu_count() - 2 # This can be dynamic
chunk_size = 1024*1024*10 # 10MB chunk

def upload_file(file_path, object_key, bucket_name, metadata=None):
    """
    Upload a file to S3 using parallel multipart upload

    Args:
        file_path: Local path to file
        object_key: S3 object key
        bucket_name: name of the bucket
        metadata: Optional metadata dictionary
    """
    file_size = os.path.getsize(file_path)
    object_key = f"vectors/{object_key}"
    try:
        # Initialize multipart upload
        response = s3_client.create_multipart_upload(
            Bucket=bucket_name,
            Key=object_key,
            Metadata=metadata or {}
        )
        upload_id = response['UploadId']

        # Calculate parts
        num_parts = math.ceil(file_size / chunk_size)
        parts = []

        # Upload parts in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for part_number in range(1, num_parts + 1):
                start_byte = (part_number - 1) * chunk_size
                end_byte = min(start_byte + chunk_size, file_size)

                futures.append(
                    executor.submit(
                        upload_part,
                        file_path,
                        bucket_name,
                        object_key,
                        upload_id,
                        part_number,
                        start_byte,
                        end_byte
                    )
                )

            # Process completed parts
            for future in futures:
                part = future.result()
                parts.append(part)

        # Complete multipart upload
        parts.sort(key=lambda x: x['PartNumber'])
        s3_client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=object_key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )

        logger.info(f"Successfully uploaded {file_path} to {object_key}")
        return True

    except Exception as e:
        logger.error(f"Error uploading file {file_path}: {str(e)}")
        # Abort multipart upload if it was initialized
        if 'upload_id' in locals():
            _abort_multipart_upload(object_key, upload_id, bucket_name)
        raise

def _abort_multipart_upload(object_key, upload_id, bucket_name):
    """Abort a multipart upload"""
    try:
        s3_client.abort_multipart_upload(
            Bucket=bucket_name,
            Key=object_key,
            UploadId=upload_id
        )
    except Exception as e:
        logger.error(f"Error aborting multipart upload: {str(e)}")

def upload_part(file_path, bucket_name, object_key, upload_id, part_number, start_byte, end_byte):
    """Upload a single part of the file"""
    client = s3_client

    retries = 3
    while retries > 0:
        try:
            with open(file_path, 'rb') as f:
                f.seek(start_byte)
                file_data = f.read(end_byte - start_byte)

            response = client.upload_part(
                Bucket=bucket_name,
                Key=object_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=file_data
            )

            return {
                'PartNumber': part_number,
                'ETag': response['ETag']
            }

        except Exception as e:
            retries -= 1
            if retries == 0:
                logger.error(
                    f"Failed to upload part {part_number} after 3 attempts: {str(e)}"
                )
                raise
            logger.warning(
                f"Retrying upload of part {part_number}. Attempts remaining: {retries}"
            )