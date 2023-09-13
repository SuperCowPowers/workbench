import os
from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_elasticache as elasticache,
    aws_logs as logs,
)
from aws_cdk.aws_ecs_patterns import ApplicationLoadBalancedFargateService
from constructs import Construct


sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")


class SageworksDashboardStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Define the ECS cluster
        cluster = ecs.Cluster(self, "SageworksCluster", vpc=ec2.Vpc(self, "SageworksVpc"))

        # Import the existing SageWorks-ExecutionRole
        sageworks_execution_role = iam.Role.from_role_arn(
            self, "ImportedSageWorksExecutionRole",
            f"arn:aws:iam::{self.account}:role/SageWorks-ExecutionRole"
        )

        # Setup CloudWatch logs
        log_group = logs.LogGroup(self, "SageWorksLogGroup")

        # Setup Redis
        redis_security_group = ec2.SecurityGroup(self, "RedisSecurityGroup", vpc=cluster.vpc)

        # Allow the ECS task to connect to the Redis cluster
        redis_security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(cluster.vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(6379)
        )

        # Create the Redis subnet group
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self,
            "RedisSubnetGroup",
            description="Subnet group for Redis",
            subnet_ids=[subnet.subnet_id for subnet in cluster.vpc.private_subnets],
        )

        # Create the Redis cluster
        redis_cluster = elasticache.CfnCacheCluster(
            self,
            "RedisCluster",
            cache_node_type="cache.t2.micro",
            engine="redis",
            num_cache_nodes=1,
            cluster_name="SageworksRedis",
            cache_subnet_group_name=redis_subnet_group.ref,
            vpc_security_group_ids=[redis_security_group.security_group_id],
        )

        # Capture the Redis endpoint
        redis_endpoint = redis_cluster.attr_redis_endpoint_address

        # Define the ECS task definition with the Docker image
        task_definition = ecs.FargateTaskDefinition(
            self,
            "SageworksTaskDef",
            task_role=sageworks_execution_role,
            memory_limit_mib=4096,
            cpu=2048,
        )

        container = task_definition.add_container(
            "SageworksContainer",
            image=ecs.ContainerImage.from_registry("public.ecr.aws/m6i5k1r2/sageworks_dashboard:latest"),
            memory_limit_mib=4096,
            environment={
                "REDIS_HOST": redis_endpoint,
                "SAGEWORKS_BUCKET": sageworks_bucket},
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="SageWorksDashboard",
                log_group=log_group)
        )
        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # Adding LoadBalancer with Fargate Service
        ApplicationLoadBalancedFargateService(
            self,
            "SageworksService",
            cluster=cluster,
            cpu=2048,
            desired_count=1,
            task_definition=task_definition,
            memory_limit_mib=4096,
            public_load_balancer=False
        )
