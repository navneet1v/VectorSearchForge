import boto3
from sagemaker import get_execution_role

sm_client = boto3.client(service_name='sagemaker')
runtime_sm_client = boto3.client(service_name='sagemaker-runtime')

account_id = boto3.client('sts').get_caller_identity()['Account']
region = "us-west-2" #boto3.Session().region_name
role = "arn:aws:iam::199552501713:role/service-role/AmazonSageMaker-ExecutionRole-20230927T185408"


from time import gmtime, strftime

time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

model_name = 'navneet-' + time_suffix

container = '{}.dkr.ecr.{}.amazonaws.com/model-endpoint-test:latest'.format(account_id, region)
instance_type = 'ml.g5.xlarge'

print('Model name: ' + model_name)
print('Container image: ' + container)

container = {
    'Image': container
}

create_model_response = sm_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    Containers = [container])

print("Model Arn: " + create_model_response['ModelArn'] + " " + model_name)

response = sm_client.update_inference_component(
        InferenceComponentName='navneet-inference-componet-3',
        Specification={
            'ModelName': model_name  # Specify the new model name here
        }
    )
print(response)

# navneet-2025-03-25-06-24-47

# aws sagemaker update-inference-component --inference-component-name navneet-inference-componet-3 --specification "{'ModelName':'navneet-2025-03-25-06-24-47','ComputeResourceRequirements': {'NumberOfAcceleratorDevicesRequired': 1.0, 'MinMemoryRequiredInMb': 1024}}"