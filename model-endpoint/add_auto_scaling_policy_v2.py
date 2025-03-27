import boto3

aas_client = boto3.client("application-autoscaling", region_name="us-west-2")
cw_client = boto3.client('cloudwatch')

def step_scaling_target_policy(endpoint_name, resource_id, max_instance_count):
    # Register scalable target with minimum capacity of 0
    aas_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=0,
        MaxCapacity=max_instance_count
    )

    # Create dynamic scale-up policy based on queue size
    scale_up_policy = aas_client.put_scaling_policy(
        PolicyName=f'DynamicScaleUp-{endpoint_name}',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='StepScaling',
        StepScalingPolicyConfiguration={
            'AdjustmentType': 'ExactCapacity', # we use exact capacity here to ensure that if due to some reason number of 
            # messages remain same we don't add extra machines
            'MetricAggregationType': 'Average',
            'Cooldown': 30,  # 30 seconds cooldown, actually it will be 60
            'StepAdjustments': [
                {
                    'MetricIntervalLowerBound': 0.0,
                    'MetricIntervalUpperBound': 2.0,
                    'ScalingAdjustment': 1  # 1-2 messages: add 1 instance
                },
                {
                    'MetricIntervalLowerBound': 2.0,
                    'MetricIntervalUpperBound': 4.0,
                    'ScalingAdjustment': 2  # 2-4 messages: add 2 instances
                },
                {
                    'MetricIntervalLowerBound': 4.0,
                    'MetricIntervalUpperBound': 6.0,
                    'ScalingAdjustment': 3  # 4-6 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 6.0,
                    'MetricIntervalUpperBound': 8.0,
                    'ScalingAdjustment': 4  # 6-8 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 8.0,
                    'MetricIntervalUpperBound': 10.0,
                    'ScalingAdjustment': 5  # 8-10 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 10.0,
                    'MetricIntervalUpperBound': 12.0,
                    'ScalingAdjustment': 6  # 10-12 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 12.0,
                    'MetricIntervalUpperBound': 14.0,
                    'ScalingAdjustment': 7  # 12-14 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 14.0,
                    'MetricIntervalUpperBound': 16.0,
                    'ScalingAdjustment': 8  # 14-16 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 16.0,
                    'MetricIntervalUpperBound': 18.0,
                    'ScalingAdjustment': 9  # 16-18 messages: add 3 instances
                },
                {
                    'MetricIntervalLowerBound': 18.0,
                    'ScalingAdjustment': max_instance_count  # >18 messages: scale to max capacity = 10
                }
            ]
        }
    )

    # Create CloudWatch alarm for dynamic scale-up based on ApproximateBacklogSize
    scale_up_alarm = cw_client.put_metric_alarm(
        AlarmName=f'DynamicScaleUpAlarm-{endpoint_name}',
        MetricName='ApproximateBacklogSize',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        Dimensions=[
            {
                'Name': 'EndpointName',
                'Value': endpoint_name
            }
        ],
        Period=30,  # 1-minute evaluation
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1.0,  # Trigger scale-up when backlog size >= 1
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing',
        AlarmActions=[scale_up_policy['PolicyARN']]
    )
    print("Scale up policy created")

    # Create gradual scale-down policy
    scale_down_policy = aas_client.put_scaling_policy(
        PolicyName=f'GradualScaleDown-{endpoint_name}',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='StepScaling',
        StepScalingPolicyConfiguration={
            'AdjustmentType': 'ChangeInCapacity',
            'MetricAggregationType': 'Average',
            'Cooldown': 100,  # Cooldown for gradual scale-down
            'StepAdjustments': [
                {
                    'MetricIntervalUpperBound': 0.0,
                    'ScalingAdjustment': -1  # Decrease by 1 instance
                }
            ]
        }
    )

    # Create CloudWatch alarm for gradual scale-down
    scale_down_alarm = cw_client.put_metric_alarm(
        AlarmName=f'GradualScaleDownAlarm-{endpoint_name}',
        MetricName='ApproximateBacklogSizePerInstance',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        Dimensions=[
            {
                'Name': 'EndpointName',
                'Value': endpoint_name
            }
        ],
        Period=120, # 2 min period, so we will get 2 datapoints in it
        EvaluationPeriods=2,
        DatapointsToAlarm=2,
        Threshold=0.0,
        ComparisonOperator='LessThanOrEqualToThreshold',
        TreatMissingData='missing',
        AlarmActions=[scale_down_policy['PolicyARN']]
    )
    print("gradual scale down to policy added")

    # Create final scale to zero policy
    scale_to_zero_policy = aas_client.put_scaling_policy(
        PolicyName=f'ScaleToZero-{endpoint_name}',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='StepScaling',
        StepScalingPolicyConfiguration={
            'AdjustmentType': 'ExactCapacity',
            'MetricAggregationType': 'Average',
            'Cooldown': 300,  # 5-minute cooldown before final scale to zero
            'StepAdjustments': [
                {
                    'MetricIntervalUpperBound': 0.0,
                    'ScalingAdjustment': 0
                }
            ]
        }
    )

    # Create alarm for final scale to zero
    scale_to_zero_alarm = cw_client.put_metric_alarm(
        AlarmName=f'ScaleToZeroAlarm-{endpoint_name}',
        MetricName='ApproximateBacklogSizePerInstance',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        Dimensions=[
            {
                'Name': 'EndpointName',
                'Value': endpoint_name
            }
        ],
        Period=300,  # 5-minute evaluation
        EvaluationPeriods=3,
        DatapointsToAlarm=3,
        Threshold=0.0,
        ComparisonOperator='LessThanOrEqualToThreshold',
        TreatMissingData='missing',
        AlarmActions=[scale_to_zero_policy['PolicyARN']]
    )

    print("scale down to 0 policy added")




current_endpoint = 98
endpoint_name = f'navneet-endpoint-async-{current_endpoint}'

endpoints = [
    {
        "name": endpoint_name
    }
]

max_instance_count = 10

for endpoint in endpoints:
    endpoint_name = endpoint["name"]
    resource_id = (
        "endpoint/" + endpoint_name + "/variant/" + "variant1"
    ) 
    step_scaling_target_policy(endpoint_name, resource_id, max_instance_count) 