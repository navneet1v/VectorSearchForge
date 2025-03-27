import boto3
from sagemaker import get_execution_role
from concurrent.futures.thread import ThreadPoolExecutor

sm_client = boto3.client(service_name='sagemaker')
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name
print(region)

role = get_execution_role()

import json
content_type = "application/json"
request_body = {"input": "This is a test with NER in America with \
    Amazon and Microsoft in Seattle, writing random stuff."}


payload = json.dumps(request_body)


def invoke_endpoint(number):
    endpoint_number = 98
    runtime_sm_client = boto3.client(service_name='sagemaker-runtime', region_name="us-west-2")
    response = runtime_sm_client.invoke_endpoint_async(
        EndpointName=f"navneet-endpoint-async-{endpoint_number}",
        InputLocation="s3://remote-index-navneet-knn/7W15UPJeTBqD-ql6yCk-4Q__3_location.s3vec.faiss.gpu",
        InvocationTimeoutSeconds=3600
    )
    #["Body"].read().decode("utf8")
    #Parse results
    output_location = response["OutputLocation"]
    print(f"Thread: {number} OutputLocation: {output_location}")

executor = ThreadPoolExecutor(max_workers=50)
for i in range(6):
    executor.submit(invoke_endpoint, i)

# {
#             "ActivityId": "0edda6bf-9288-4295-b67f-d78c3be70152",
#             "ServiceNamespace": "sagemaker",
#             "ResourceId": "endpoint/navneet-endpoint-async-5/variant/variant1",
#             "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
#             "Description": "Setting desired instance count to 48.",
#             "Cause": "monitor alarm TargetTracking-endpoint/navneet-endpoint-async-5/variant/variant1-AlarmHigh-a9ace065-59c2-4875-a224-6ca6acdb15b3 in state ALARM triggered policy Invocations-ScalingPolicy-navneet-endpoint-async-5",
#             "StartTime": "2025-03-26T02:28:13.498000+00:00",
#             "EndTime": "2025-03-26T02:35:19.187000+00:00",
#             "StatusCode": "Successful",
#             "StatusMessage": "Successfully set desired instance count to 48. Change successfully fulfilled by sagemaker."
#         },