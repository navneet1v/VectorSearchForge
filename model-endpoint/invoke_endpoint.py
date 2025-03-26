import boto3
from sagemaker import get_execution_role

runtime_sm_client = boto3.client(service_name='sagemaker-runtime', region_name="us-west-2")
sm_client = boto3.client(service_name='sagemaker')
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name
print(region)

#used to store model artifacts which SageMaker AI will extract to /opt/ml/model in the container,
#in this example case we will not be making use of S3 to store the model artifacts
#s3_bucket = '<S3Bucket>'

role = get_execution_role()

import json
content_type = "application/json"
request_body = {"input": "This is a test with NER in America with \
    Amazon and Microsoft in Seattle, writing random stuff."}

#Serialize data for endpoint
#data = json.loads(json.dumps(request_body))
payload = json.dumps(request_body)

#Endpoint invocation
# response = runtime_sm_client.invoke_endpoint(
#     EndpointName="navneet-endpoint-0",
#     ContentType=content_type,
#     Body=payload)

endpoint_number = 4 

desc = sm_client.describe_inference_component(InferenceComponentName=f"navneet-inference-componet-{endpoint_number}")
print(desc)


response = runtime_sm_client.invoke_endpoint(
    EndpointName=f"navneet-endpoint-{endpoint_number}",
    InferenceComponentName=f'navneet-inference-componet-{endpoint_number}',
    Body=json.dumps(request_body),
    ContentType="application/json",
)
#["Body"].read().decode("utf8")
#Parse results
result = json.loads(response['Body'].read().decode())
print(result)