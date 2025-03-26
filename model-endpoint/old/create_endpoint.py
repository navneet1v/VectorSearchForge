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

print("Model Arn: " + create_model_response['ModelArn'])

max_endpoints = 5
current_endpoint = 4
max_instance_count = 2
endpoints = []
while current_endpoint < max_endpoints:
    endpoint_config_name = f'navneet-endpoint-config-{current_endpoint}'
    print('Endpoint config name: ' + endpoint_config_name)

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ExecutionRoleArn=role,
        ProductionVariants=[{
            'InstanceType': instance_type,
            "ManagedInstanceScaling": {
                "Status": "ENABLED",
                "MinInstanceCount": 0,
                "MaxInstanceCount": max_instance_count,
            },
            'InitialInstanceCount': 1,
            #'InitialVariantWeight': 1,
            #'ModelName': model_name, when using inference component this is not needed
            'VariantName': 'AllTraffic',
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
            }]
    )

    print("Endpoint config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

    endpoint_name = f'navneet-endpoint-{current_endpoint}'
    
    print('Endpoint name: ' + endpoint_name)

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

    inference_component_name = f'navneet-inference-componet-{current_endpoint}'

    sm_client.create_inference_component(
        InferenceComponentName=inference_component_name,
        EndpointName=endpoint_name,
        VariantName='AllTraffic',
        Specification={
            "ModelName": model_name,
            "StartupParameters": {
                "ModelDataDownloadTimeoutInSeconds": 3600,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 1024, # 2GB, the instance has 16GB
                "NumberOfAcceleratorDevicesRequired": 1
            },
        },
        RuntimeConfig={
            "CopyCount": 1,
        },
        # RuntimeConfig={
        #     'InstanceCount': 1,  # Initial instance count
        #     'MaxConcurrency': 5  # Maximum number of concurrent requests per instance
        # }
    )

    print(f"Inference component name: {inference_component_name}")

    endpoints.append({"name": endpoint_name, "arn": create_endpoint_response['EndpointArn'], "inference_component": inference_component_name})
    current_endpoint = current_endpoint + 1

for endpoint in endpoints:
    resp = sm_client.describe_endpoint(EndpointName=endpoint['name'])
    status = resp['EndpointStatus']
    print("Endpoint Status: " + status)

    print('Waiting for {} endpoint to be in service...'.format(endpoint['name']))
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint['name'])

print("Now creating autoscaling policy")
# Lets create aas client
aas_client = boto3.client("application-autoscaling", region_name="us-west-2")

# Now lets create the scaling policy
for endpoint in endpoints:
    inference_component_name = endpoint["inference_component"]
    # Register scalable target
    resource_id = f"inference-component/{inference_component_name}"
    service_namespace = "sagemaker"
    scalable_dimension = "sagemaker:inference-component:DesiredCopyCount"
    # this for scaling down to 0
    aas_client.register_scalable_target(
        ServiceNamespace=service_namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dimension,
        MinCapacity=0,
        MaxCapacity=max_instance_count,  # Replace with your desired maximum number of model copies
    )

    # scale up when innovcation request goes above 5.
    aas_client.put_scaling_policy(
        PolicyName=f"inference-component-target-tracking-scaling-policy-{endpoint['name']}",
        PolicyType="TargetTrackingScaling",
        ServiceNamespace=service_namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dimension,
        TargetTrackingScalingPolicyConfiguration={
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution",
            },
            # Low TPS + load TPS
            "TargetValue": 5,  # you need to adjust this value based on your use case
            "ScaleInCooldown": 600, # scale down in 5 mins.
            "ScaleOutCooldown": 60,  
        },
    )

    # Scale out from 0 to a desiered number
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/application-autoscaling/client/put_scaling_policy.html
    scaleout_policy = aas_client.put_scaling_policy(
        PolicyName=f"inference-component-step-scaling-policy-{endpoint['name']}",
        PolicyType="StepScaling",
        ServiceNamespace=service_namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dimension,
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": 60,
            "StepAdjustments":
            [
                {
                    "MetricIntervalLowerBound": 0,
                    "ScalingAdjustment": 1 # you need to adjust this value based on your use case
                }
            ]
        },
    )

    print(f"Scale out policy {scaleout_policy}")

    cw_client = boto3.client('cloudwatch')
    # this is for cold start
    cw_client.put_metric_alarm(
        AlarmName=f"ic-step-scaling-policy-alarm-{endpoint['name']}",
        AlarmActions=[scaleout_policy["PolicyARN"]],  # Replace with your actual ARN
        MetricName='NoCapacityInvocationFailures', # this is metrics to scale on
        Namespace='AWS/SageMaker',
        Statistic='Maximum',
        Dimensions=[
            {
                'Name': 'InferenceComponentName',
                'Value': inference_component_name 
            }
        ],
        Period=30,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1,
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing'
    )



