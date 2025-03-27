import boto3
from sagemaker import get_execution_role

sm_client = boto3.client(service_name='sagemaker')
runtime_sm_client = boto3.client(service_name='sagemaker-runtime')

account_id = boto3.client('sts').get_caller_identity()['Account']
region = "us-west-2" #boto3.Session().region_name
role = "arn:aws:iam::199552501713:role/service-role/AmazonSageMaker-ExecutionRole-20230927T185408"


from time import gmtime, strftime

time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

model_name = 'navneet-async' + time_suffix

container = '{}.dkr.ecr.{}.amazonaws.com/model-endpoint-test:latest'.format(account_id, region)
instance_type = 'ml.g5.2xlarge'

print('Model name: ' + model_name)
print('Container image: ' + container)

container = {
    'Image': container,
     "Environment": { 
        "IS_ASYNC" : "1",
        "MODEL_SERVER_TIMEOUT": "3600",
        "MODEL_SERVER_WORKERS": "2"
        }
}

create_model_response = sm_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    Containers = [container])

print("Model Arn: " + create_model_response['ModelArn'])

max_endpoints = 99
current_endpoint = 98
max_instance_count = 10
endpoints = [{"name": f"navneet-endpoint-async-{current_endpoint}"}]
while current_endpoint < max_endpoints:
    endpoint_config_name = f'navneet-endpoint-config-async-{current_endpoint}'
    print('Endpoint config name: ' + endpoint_config_name)

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://remote-index-navneet-knn",
            },
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 2},
        },
    )

    print(f"Created EndpointConfig: {create_endpoint_config_response['EndpointConfigArn']}")

    endpoint_name = f'navneet-endpoint-async-{current_endpoint}'
    
    print('Endpoint name: ' + endpoint_name)

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

    endpoints.append({"name": endpoint_name, "arn": create_endpoint_response['EndpointArn']})
    current_endpoint = current_endpoint + 1

for endpoint in endpoints:
    resp = sm_client.describe_endpoint(EndpointName=endpoint['name'])
    status = resp['EndpointStatus']
    print("Endpoint Status: " + status)

    print('Waiting for {} endpoint to be in service...'.format(endpoint['name']))
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint['name'])

# print("Now creating autoscaling policy")
# # Lets create aas client
# aas_client = boto3.client("application-autoscaling", region_name="us-west-2")

# # Now lets create the scaling policy
# for endpoint in endpoints:
#     endpoint_name = endpoint["name"]
#     resource_id = (
#         "endpoint/" + endpoint_name + "/variant/" + "variant1"
#     )  

#     # Configure Autoscaling on asynchronous endpoint down to zero instances
#     response = aas_client.register_scalable_target(
#         ServiceNamespace="sagemaker",
#         ResourceId=resource_id,
#         ScalableDimension="sagemaker:variant:DesiredInstanceCount",
#         MinCapacity=0,
#         MaxCapacity=max_instance_count,
#     )

#     response = aas_client.put_scaling_policy(
#         PolicyName=f"Invocations-ScalingPolicy-{endpoint_name}",
#         ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
#         ResourceId=resource_id,  # Endpoint name
#         ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
#         # PolicyType="StepScaling",  # 'StepScaling'|'TargetTrackingScaling' # step scaling policy require a CW metrics
#         # StepScalingPolicyConfiguration={
#         #     "AdjustmentType": "ChangeInCapacity",
#         #     "MetricAggregationType": "Maximum",
#         #     "Cooldown": 60,
#         #     "StepAdjustments":
#         #     [
#         #         {
#         #             "MetricIntervalLowerBound": 0,
#         #             "ScalingAdjustment": 1 # you need to adjust this value based on your use case
#         #         }
#         #     ]
#         # },
#         PolicyType="TargetTrackingScaling",  # 'StepScaling'|'TargetTrackingScaling'
#         TargetTrackingScalingPolicyConfiguration={
#             "TargetValue": 5.0,  # The target value for the metric. - here the metric is - SageMakerVariantInvocationsPerInstance
#             "CustomizedMetricSpecification": {
#                 "MetricName": "ApproximateBacklogSizePerInstance", #ApproximateBacklogSize or ApproximateBacklogSizePerInstance
#                 "Namespace": "AWS/SageMaker",
#                 "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
#                 "Statistic": "Average",
#             },
#             "ScaleInCooldown": 30,  # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
#             # additional instances before the effects of previous activities are visible.
#             # You can configure the length of time based on your instance startup time or other application needs.
#             # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
#             "ScaleOutCooldown": 30,  # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
#             'DisableScaleIn': False #- indicates whether scale in by the target tracking policy is disabled.
#             # If the value is true , scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
#         },
#     )


