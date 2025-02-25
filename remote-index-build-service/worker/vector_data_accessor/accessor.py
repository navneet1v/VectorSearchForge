from dataclasses import dataclass

import numpy as np
from models.data_model import CreateIndexRequest
import s3.s3client as s3
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorsDataset:
    vectors: np.ndarray
    ids: np.array
    dimensions: int

    def free_vectors_space(self):
        del self.vectors
        del self.ids

    @staticmethod
    def get_vector_dataset(createIndexRequest: CreateIndexRequest):
        if not s3.check_s3_object_exists(createIndexRequest.bucketName, createIndexRequest.objectLocation):
            raise TypeError(f"{createIndexRequest.objectLocation} does not exist in the bucket : {createIndexRequest.bucketName}")
        vector_file = s3.download_s3_file_in_chunks(createIndexRequest.bucketName, createIndexRequest.objectLocation)
        ids_file_object_location = f"{createIndexRequest.objectLocation.split('.')[0]}.knndid"
        ids_file = s3.download_s3_file_in_chunks(createIndexRequest.bucketName, ids_file_object_location)
        vector_dataset = VectorsDataset.__parse(vector_file, createIndexRequest.dimensions, createIndexRequest.numberOfVectors, ids_file)
        logger.info(f"Deleting the donwnloaded file {vector_file}")
        os.remove(vector_file)
        return vector_dataset

    @staticmethod
    def __parse(vector_file: str, dimension: int, number_of_vectors: int, ids_file: str, id_dtype: str = '<i8', vector_dtype: str = '<f4'):
        """
        Parse binary vector data from a file into a VectorsDataset object.

        This private static method reads binary data containing vectors and their IDs
        from a file. The data is expected to be in little-endian format, typically
        written by Java applications.

        Args:
            vector_file (str): Path to the binary file containing vector data.
            dimension (int): Number of dimensions for each vector.
            number_of_vectors (int): Total number of vectors to read from the file.
            ids_file (str): path to the binary file containing ids
            id_dtype (str, optional): NumPy dtype for reading IDs.
                Defaults to '<i8' (little-endian 64-bit integer).
            vector_dtype (str, optional): NumPy dtype for reading vector values.
                Defaults to '<f4' (little-endian 32-bit float).

        Returns:
            VectorsDataset: A new instance containing the parsed vectors and their IDs.

        Raises:
            ValueError: If the number of values read doesn't match the expected size
                (number_of_vectors * dimension).
            IOError: If there are issues reading the file.
            TypeError: If the file content doesn't match the expected data types.

        Example:
            >>> dataset = VectorsDataset.__parse(
            ...     'vectors.bin',
            ...     dimension=128,
            ...     number_of_vectors=1000,
            ...     id_dtype='>i8',
            ...     vector_dtype='>f4'
            ... )

        Notes:
            - The binary file should contain vectors in a contiguous block
            - Data is expected to be in little-endian format ('<') for Java compatibility
            - Supported vector types:
                * '<f4': 32-bit float (Java float)
                * '<f8': 64-bit float (Java double)
            - Supported ID types:
                * '<i4': 32-bit integer (Java int)
                * '<i8': 64-bit integer (Java long)

        Memory Usage:
            The method will allocate memory for:
            - Vector array: number_of_vectors * dimension * sizeof(vector_dtype)
            - ID array: number_of_vectors * sizeof(id_dtype)
        """
        vectors = None

        with open(ids_file, 'rb') as f:
            ids = np.fromfile(f, dtype=np.int32, count=number_of_vectors)
            if len(ids) != number_of_vectors:
                raise ValueError(
                    f"Expected ids to be {number_of_vectors} values, "
                    f"but got {len(ids)}"
                )

        with open(vector_file, 'rb') as f:
            vectors = np.fromfile(f, dtype=vector_dtype,
                                  count=number_of_vectors * dimension)

            if len(vectors) != number_of_vectors * dimension:
                raise ValueError(
                    f"Expected {number_of_vectors * dimension} values, "
                    f"but got {len(vectors)}"
                )

            # Reshape the vectors array
            vectors = vectors.reshape(number_of_vectors, dimension)
        return VectorsDataset(vectors=vectors, dimensions=dimension, ids=ids)
