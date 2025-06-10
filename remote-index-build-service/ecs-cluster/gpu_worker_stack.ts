import autoscaling = require("aws-cdk-lib/aws-autoscaling");
import ec2 = require("aws-cdk-lib/aws-ec2");
import ecs = require("aws-cdk-lib/aws-ecs");
import cdk = require("aws-cdk-lib");
import {NetworkMode, Protocol} from "aws-cdk-lib/aws-ecs";
import {Size} from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import {SecurityGroup} from "aws-cdk-lib/aws-ec2";
import {ApplicationLoadBalancer} from "aws-cdk-lib/aws-elasticloadbalancingv2";
import {Construct} from "constructs";
import {CommonStackProps} from "./index";
import {EbsDeviceVolumeType} from "aws-cdk-lib/aws-autoscaling";

export const SERVICE_PREFIX:string = "vector-index-build-service"
export const WORKER_IMAGE_NAME: string = "worker-image";

export interface GpuWorkerStackProps extends CommonStackProps {
    readonly loadBalancer: ApplicationLoadBalancer
}

/**
 * This stack creates a fleet of GPU instances that can be used to build vector indexes.
 * The instances are launched using an ECS cluster and an AutoScalingGroup.
 */
export class GpuWorkerStack extends cdk.Stack {
    private workerImage: string;

    private initClassParams(scope: cdk.App) {
        this.workerImage = `${scope.node.tryGetContext(WORKER_IMAGE_NAME)}`;
    }

    private getVPC(scope: Construct, props: GpuWorkerStackProps) {
        return ec2.Vpc.fromLookup(scope, props.vpc, {
            vpcId: props.vpc
        });
    }

    constructor(scope: cdk.App, id: string, props: GpuWorkerStackProps) {
        super(scope, id, props);
        this.initClassParams(scope);
        const vpc = this.getVPC(this, props);

        // Autoscaling group that will launch a fleet of instances that have GPU's
        const asg = new autoscaling.AutoScalingGroup(this, id + "-asg", {
            instanceType: ec2.InstanceType.of(
                ec2.InstanceClass.G5,
                ec2.InstanceSize.XLARGE2
            ),
            machineImage: ec2.MachineImage.fromSsmParameter(
                "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
            ),
            desiredCapacity: 3,
            vpc,
            maxCapacity: 10,
            securityGroup: SecurityGroup.fromLookupById(this, id + "-sg", props.securityGroup),
            blockDevices: [
                {
                    deviceName: '/dev/xvda', // Root volume
                    volume: autoscaling.BlockDeviceVolume.ebs(1000, {
                        volumeType: EbsDeviceVolumeType.GP3,
                        deleteOnTermination: true
                    }),
                }
            ]

        });

        // Attach the fleet to an ECS cluster with a capacity provider.
        // This capacity provider will automatically scale up the ASG
        // to launch more GPU instances when GPU tasks need them.
        const cluster = new ecs.Cluster(this, id + "EcsCluster", { vpc: vpc, containerInsights: true});
        const capacityProvider = new ecs.AsgCapacityProvider(
            this,
            id + "AsgCapacityProvider",
            { autoScalingGroup: asg }
        );
        cluster.addAsgCapacityProvider(capacityProvider);


        // Define a task that requires GPU.
        const gpuTaskDefinition = new ecs.Ec2TaskDefinition(this, id + "gpu-task" , {
            networkMode: NetworkMode.HOST, // This will ensure that we are able to access the IP of the machine
            family: "gpu-worker-task-definition",
            taskRole: iam.Role.fromRoleArn(this, id + "gpu-task-role", props.taskRole),
            executionRole: iam.Role.fromRoleArn(this, id + "gpu-task-execution", props.executionRole),
            volumes: [
                {
                    "name": "worker-volume",
                    "host": {}
                }
            ]
        });


        gpuTaskDefinition.addContainer( id + "-gpu", {
            containerName:  "worker-gpu-container",
            essential: true,
            image: new ecs.RepositoryImage(this.workerImage),

            gpuCount: 1,
            cpu: 8 * 1024,
            memoryLimitMiB: 30 * 1024, // hard limit, max we have is 32
            memoryReservationMiB: 30 * 1024, // soft limit
            portMappings: [
                {
                    containerPort: 6005,
                    hostPort: 6005,
                    protocol: Protocol.TCP
                }
            ],
            environment: {
                "COORDINATOR_NODE_URL": `${props?.loadBalancer.loadBalancerDnsName}`,
                "PYTHONFAULTHANDLER": "1",
                "REGISTER_WITH_COORDINATOR": "1",
                "COORDINATOR_NODE_PROTOCOL": "http",
                "COORDINATOR_NODE_PORT": "80",
                "INDEX_BUILD_TYPE": "gpu"
            },
            workingDirectory: "/remote_vector_index_builder",
            user: "root",
            ulimits: [
                {
                    name: ecs.UlimitName.CORE,
                    softLimit: -1,
                    hardLimit: -1,
                },
            ],
            logging: new ecs.AwsLogDriver({
                streamPrefix: "gpu-service",
                mode: ecs.AwsLogDriverMode.NON_BLOCKING,
                maxBufferSize: Size.mebibytes(25),
                logGroup: new cdk.aws_logs.LogGroup(this, id + "gpu-service-loggroup", {
                    logGroupName: "/ecs/gpu-service-loggroup",
                    removalPolicy: cdk.RemovalPolicy.DESTROY,
                })

            }),
        }).addMountPoints({
            containerPath: "/app/logs",
            readOnly: false,
            sourceVolume: "worker-volume"
        });


        // Request ECS to launch the task onto the fleet
        new ecs.Ec2Service(this, id + "gpu-service", {
            cluster,
            desiredCount: 3,
            // Service will automatically request capacity from the
            // capacity provider
            capacityProviderStrategies: [
                {
                    capacityProvider: capacityProvider.capacityProviderName,
                    base: 0,
                    weight: 1,
                },
            ],
            taskDefinition: gpuTaskDefinition
        });
    }
}