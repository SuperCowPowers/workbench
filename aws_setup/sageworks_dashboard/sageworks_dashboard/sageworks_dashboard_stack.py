from typing import Optional, List
from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_elasticache as elasticache,
    aws_logs as logs,
)
from aws_cdk.aws_ec2 import Subnet
from aws_cdk.aws_ecs_patterns import ApplicationLoadBalancedFargateService
from constructs import Construct


class SageworksDashboardStackProps:
    def __init__(self, sageworks_bucket: str, existing_vpc_id: Optional[str] = None,
                 existing_subnet_ids: Optional[List[str]] = None,
                 whitelist_ips: Optional[List[str]] = None):
        self.sageworks_bucket = sageworks_bucket
        self.existing_vpc_id = existing_vpc_id
        self.existing_subnet_ids = existing_subnet_ids
        self.whitelist_ips = whitelist_ips


class SageworksDashboardStack(Stack):
    def __init__(self, scope: Construct, id: str, props: SageworksDashboardStackProps, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        if not props.sageworks_bucket:
            raise ValueError("sageworks_bucket is a required property")

        # Create a cluster using a EXISTING VPC
        if props.existing_vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "ImportedVPC", vpc_id=props.existing_vpc_id)
            cluster = ecs.Cluster(self, "SageworksCluster", vpc=vpc)

        # Create a cluster using a NEW VPC
        else:
            cluster = ecs.Cluster(self, "SageworksCluster", vpc=ec2.Vpc(self, "SageworksVpc", max_azs=2))

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
                "SAGEWORKS_BUCKET": props.sageworks_bucket},
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="SageWorksDashboard",
                log_group=log_group)
        )
        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # Create security group for the load balancer
        lb_security_group = ec2.SecurityGroup(self, "LoadBalancerSecurityGroup", vpc=cluster.vpc)

        # Add rules for the whitelist IPs
        if props.whitelist_ips:
            for ip in props.whitelist_ips:
                lb_security_group.add_ingress_rule(
                    ec2.Peer.ipv4(ip),
                    ec2.Port.tcp(80)
                )

        # Were we given a subnet selection?
        subnet_selection = None
        if props.existing_subnet_ids:
            subnets = [ec2.Subnet.from_subnet_id(self, f"Subnet{i}", subnet_id) for i, subnet_id in enumerate(props.existing_subnet_ids)]
            print(subnets)
            subnet_selection = ec2.SubnetSelection(subnets=subnets)
            print(subnet_selection)

        # Adding LoadBalancer with Fargate Service
        # TODO: Add logic to use existing subnets
        fargate_service = ApplicationLoadBalancedFargateService(
            self,
            "SageworksService",
            cluster=cluster,
            cpu=2048,
            desired_count=1,
            task_definition=task_definition,
            memory_limit_mib=4096,
            public_load_balancer=True,
            security_groups=[lb_security_group],
            open_listener=False,
            # task_subnets=subnet_selection
        )

        # Remove all default security group from the load balancer
        fargate_service.load_balancer.connections.security_groups.clear()

        # Add our custom security group
        fargate_service.load_balancer.add_security_group(lb_security_group)
