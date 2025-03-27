import boto3

aas_client = boto3.client("application-autoscaling", region_name="us-west-2")
cw_client = boto3.client('cloudwatch')

def target_tracking_scaling_policy(endpoint_name, resource_id, max_instance_count):
    # Configure Autoscaling on asynchronous endpoint down to zero instances
    response = aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=max_instance_count,
    )


    # How target Tracking Scaling works: https://repost.aws/questions/QUT4xru2SdTSqtoNyt1XV3VA/configuring-auto-scaling-for-sagemaker-async-inference#ANX-4rH_AUQIqBpHyb4M5lrA
    response = aas_client.put_scaling_policy(
        PolicyName=f"Invocations-ScalingPolicy-{endpoint_name}",
        ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        # PolicyType="StepScaling",  # 'StepScaling'|'TargetTrackingScaling' # step scaling policy require a CW metrics
        # StepScalingPolicyConfiguration={
        #     "AdjustmentType": "ChangeInCapacity",
        #     "MetricAggregationType": "Maximum",
        #     "Cooldown": 60,
        #     "StepAdjustments":
        #     [
        #         {
        #             "MetricIntervalLowerBound": 0,
        #             "ScalingAdjustment": 1 # you need to adjust this value based on your use case
        #         }
        #     ]
        # },
        PolicyType="TargetTrackingScaling",  # 'StepScaling'|'TargetTrackingScaling'
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 5,  # The target value for the metric. - here the metric is - SageMakerVariantInvocationsPerInstance
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance", #ApproximateBacklogSize or ApproximateBacklogSizePerInstance
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 30,  # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
            # additional instances before the effects of previous activities are visible.
            # You can configure the length of time based on your instance startup time or other application needs.
            # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
            "ScaleOutCooldown": 30,  # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
            'DisableScaleIn': False #- indicates whether scale in by the target tracking policy is disabled.
            # If the value is true , scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
        },
    )

    print(f"Auto scaling policy for scale up added : {endpoint_name}")

    # adding policy to scale from 0.
    response = aas_client.put_scaling_policy(
        PolicyName="HasBacklogWithoutCapacity-ScalingPolicy",
        ServiceNamespace="sagemaker",  # The namespace of the service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="StepScaling",  # 'StepScaling' or 'TargetTrackingScaling'
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity", # Specifies whether the ScalingAdjustment value in the StepAdjustment property is an absolute number or a percentage of the current capacity. 
            "MetricAggregationType": "Average", # The aggregation type for the CloudWatch metrics.
            "Cooldown": 30, # The amount of time, in seconds, to wait for a previous scaling activity to take effect. 
            "StepAdjustments": # A set of adjustments that enable you to scale based on the size of the alarm breach.
            [
                {
                    "MetricIntervalLowerBound": 0,
                    "ScalingAdjustment": 1
                }
            ]
        },    
    )
    
    cw_client = boto3.client('cloudwatch')

    response = cw_client.put_metric_alarm(
        AlarmName=f"scaling-from-zero-{endpoint['name']}",
        MetricName='HasBacklogWithoutCapacity',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        EvaluationPeriods= 1,
        DatapointsToAlarm= 1,
        Threshold= 1,
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing',
        Dimensions=[
            { 'Name':'EndpointName', 'Value':endpoint_name },
        ],
        Period= 60,
        AlarmActions=[response["PolicyARN"]],  # Replace with your actual ARN
    )

    print(f"Scale to zero metrics also added for {endpoint_name}")


    # aws application-autoscaling describe-scaling-activities   --service-namespace sagemaker


def step_scaling_target_policy(endpoint_name, resource_id, max_instance_count):
    # Configure Autoscaling on asynchronous endpoint down to zero instances
    aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=max_instance_count,
    )


    # How target Tracking Scaling works: https://repost.aws/questions/QUT4xru2SdTSqtoNyt1XV3VA/configuring-auto-scaling-for-sagemaker-async-inference#ANX-4rH_AUQIqBpHyb4M5lrA
    response = aas_client.put_scaling_policy(
        PolicyName=f"Invocations-ScalingPolicy-{endpoint_name}",
        ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="StepScaling",  # 'StepScaling'|'TargetTrackingScaling' # step scaling policy require a CW metrics
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": 30,
            "StepAdjustments":
            [ 
                # This to scale to 0. we should move this as a separate policy.
                # {
                #     "MetricIntervalUpperBound": 0.9,
                #     "MetricIntervalLowerBound": 0.0,
                #     "ScalingAdjustment": 0
                # },
                # this is for scaling up
                {
                    "MetricIntervalLowerBound": 0.9,
                    "MetricIntervalUpperBound": 2.1,
                    "ScalingAdjustment": 1
                },
                {
                    "MetricIntervalLowerBound": 2.1,
                    "MetricIntervalUpperBound": 4.1,
                    "ScalingAdjustment": 2
                },
                {
                    "MetricIntervalLowerBound": 4.1,
                    "MetricIntervalUpperBound": 6.1,
                    "ScalingAdjustment": 3
                },
                {
                    "MetricIntervalLowerBound": 6.1,
                    "MetricIntervalUpperBound": 8.1,
                    "ScalingAdjustment": 4
                },
                {
                    "MetricIntervalLowerBound": 8.1,
                    "ScalingAdjustment": 10
                }
            ]
        },
    )


    response = cw_client.put_metric_alarm(
        AlarmName=f"combined-scaling-up-policy-{endpoint['name']}",
        Metrics=[
            {
                "Label": "ApproximateBacklogSizePerInstance",
                "Id": "m1",
                "MetricStat": {
                    "Metric": {
                        "MetricName": "ApproximateBacklogSizePerInstance",
                        "Namespace": "AWS/SageMaker",
                        "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                    },
                    "Stat": "Average",
                    "Period": 60
                },
                "ReturnData": False
            },
            {
                "Label": "Get the ECS running task count (the number of currently running tasks)",
                "Id": "m2",
                "MetricStat": {
                    "Metric": {
                        "MetricName": "HasBacklogWithoutCapacity",
                        "Namespace": "AWS/SageMaker",
                        "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                    },
                    "Period": 60,
                    "Stat": "Average"
                },
                "ReturnData": False
            },
            {
                "Label": "Calculate should we do scaling or not",
                "Id": "e1",
                "Expression": "MAX([m1, m2])",
                "ReturnData": True
            }
        ],
        #Statistic='Average',
        EvaluationPeriods= 1,
        DatapointsToAlarm= 1,
        Threshold= 0.9,
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing',
        # Dimensions=[
        #     { 'Name':'EndpointName', 'Value':endpoint_name },
        # ],
        #Period= 60,
        AlarmActions=[response["PolicyARN"]],  # Replace with your actual ARN
    )
    print(f"Step Scaling Auto scaling policy for scale up and scale down added : {endpoint_name}")

    # lets write scale down policy
    response = aas_client.put_scaling_policy(
        PolicyName=f"Invocations-ScalingPolicy-down-{endpoint_name}",
        ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="StepScaling",  # 'StepScaling'|'TargetTrackingScaling' # step scaling policy require a CW metrics
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": 30,
            "StepAdjustments":
            [ 
                # This to scale to 0. we should move this as a separate policy.
                {
                    #"MetricIntervalUpperBound": 0.9,
                    "MetricIntervalLowerBound": 0.0,
                    "ScalingAdjustment": 0
                }
            ]
        },
    )

    response = cw_client.put_metric_alarm(
        AlarmName=f"scaling-to-zero-{endpoint['name']}",
        MetricName='ApproximateBacklogSize',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        EvaluationPeriods= 10,
        DatapointsToAlarm= 10,
        Threshold= 0,
        ComparisonOperator='LessThanOrEqualToThreshold',
        TreatMissingData='missing',
        Dimensions=[
            { 'Name':'EndpointName', 'Value':endpoint_name },
        ],
        Period= 60,
        AlarmActions=[response["PolicyARN"]],  # Replace with your actual ARN
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