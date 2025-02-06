from typing import Optional, List
from aws_cdk import (
    Stack,
    StackProps,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_elasticache as elasticache,
    aws_logs as logs,
)
from aws_cdk.aws_certificatemanager import Certificate
from aws_cdk.aws_ecs_patterns import ApplicationLoadBalancedFargateService
from constructs import Construct


class CompoundExplorerStackProps(StackProps):
    def __init__(
        self,
        dashboard_image: str,
        workbench_bucket: str,
        workbench_api_key: str,
        workbench_plugins: str,
        existing_vpc_id: Optional[str] = None,
        existing_subnet_ids: Optional[List[str]] = None,
        whitelist_ips: Optional[List[str]] = None,
        whitelist_prefix_lists: Optional[List[str]] = None,
        certificate_arn: Optional[str] = None,
        public: bool = False,
    ):
        self.dashboard_image = dashboard_image
        self.workbench_bucket = workbench_bucket
        self.workbench_api_key = workbench_api_key
        self.workbench_plugins = workbench_plugins
        self.existing_vpc_id = existing_vpc_id
        self.existing_subnet_ids = existing_subnet_ids
        self.whitelist_ips = whitelist_ips
        self.certificate_arn = certificate_arn
        self.whitelist_prefix_lists = whitelist_prefix_lists
        self.public = public


class CompoundExplorerStack(Stack):
    def __init__(self, scope: Construct, id: str, props: CompoundExplorerStackProps, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        if not props.workbench_bucket:
            raise ValueError("workbench_bucket is a required property")

        # Create a cluster using a EXISTING VPC
        if props.existing_vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "ImportedVPC", vpc_id=props.existing_vpc_id)
            cluster = ecs.Cluster(self, "ExplorerCluster", vpc=vpc)

        # Create a cluster using a NEW VPC
        else:
            vpc = ec2.Vpc(self, "ExplorerVpc", max_azs=2)
            cluster = ecs.Cluster(self, "ExplorerCluster", vpc=vpc)

            # Add Network ACL rules for the Redis cluster
            self._configure_network_acls(vpc, props.whitelist_ips)

        # Import the existing Workbench-ExecutionRole
        workbench_execution_role = iam.Role.from_role_arn(
            self, "ImportedWorkbenchExecutionRole", f"arn:aws:iam::{self.account}:role/Workbench-ExecutionRole"
        )

        # Setup CloudWatch logs
        log_group = logs.LogGroup(self, "ExplorerLogGroup")

        # Setup Security Group for Redis
        redis_security_group = ec2.SecurityGroup(self, "ExplorerRedisSecurityGroup", vpc=cluster.vpc)

        # Allow the ECS task to connect to the Redis cluster
        redis_security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(cluster.vpc.vpc_cidr_block), connection=ec2.Port.tcp(6379)
        )

        # Allow specific whitelist IPs to connect to the Redis cluster
        if props.whitelist_ips:
            for ip in props.whitelist_ips:
                redis_security_group.add_ingress_rule(ec2.Peer.ipv4(ip), ec2.Port.tcp(6379))

        # Adding AWS Managed Prefix Lists to connect to the Redis cluster
        if props.whitelist_prefix_lists:
            for pl in props.whitelist_prefix_lists:
                redis_security_group.add_ingress_rule(ec2.Peer.prefix_list(pl), ec2.Port.tcp(6379))

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
            cluster_name="ExplorerRedis",
            cache_subnet_group_name=redis_subnet_group.ref,
            vpc_security_group_ids=[redis_security_group.security_group_id],
        )

        # Capture the Redis endpoint
        redis_endpoint = redis_cluster.attr_redis_endpoint_address

        # Define the ECS task definition with the Docker image
        task_definition = ecs.FargateTaskDefinition(
            self,
            "ExplorerTaskDef",
            task_role=workbench_execution_role,
            memory_limit_mib=4096,
            cpu=1024,
        )
        container = task_definition.add_container(
            "ExplorerContainer",
            image=ecs.ContainerImage.from_registry(props.dashboard_image),
            memory_limit_mib=4096,
            environment={
                "REDIS_HOST": redis_endpoint,
                "WORKBENCH_BUCKET": props.workbench_bucket,
                "WORKBENCH_API_KEY": props.workbench_api_key,
                "WORKBENCH_PLUGINS": props.workbench_plugins,
            },
            logging=ecs.LogDriver.aws_logs(stream_prefix="CompoundExplorer", log_group=log_group),
        )
        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # Create a NEW Security Group for the Load Balancer
        lb_security_group = ec2.SecurityGroup(self, "LoadBalancerSecurityGroup", vpc=cluster.vpc)

        # Add rules for the whitelist IPs
        if props.whitelist_ips:
            for ip in props.whitelist_ips:
                lb_security_group.add_ingress_rule(ec2.Peer.ipv4(ip), ec2.Port.tcp(443))

        # Adding AWS Managed Prefix Lists
        if props.whitelist_prefix_lists:
            for pl in props.whitelist_prefix_lists:
                lb_security_group.add_ingress_rule(ec2.Peer.prefix_list(pl), ec2.Port.tcp(443))

        # Import existing SSL certificate if certificate_arn is provided
        certificate = (
            Certificate.from_certificate_arn(self, "ImportedCertificate", certificate_arn=props.certificate_arn)
            if props.certificate_arn
            else None
        )

        # Allow HTTP/HTTPS access from the internet (public)
        if props.public:
            if certificate:
                # Only allow HTTPS access if a certificate is provided
                lb_security_group.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443))
            else:
                # Allow HTTP access if no certificate is provided
                lb_security_group.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80))

        # Adding LoadBalancer with Fargate Service
        fargate_service = ApplicationLoadBalancedFargateService(
            self,
            "ExplorerService",
            cluster=cluster,
            cpu=1024,
            desired_count=1,
            task_definition=task_definition,
            memory_limit_mib=4096,
            public_load_balancer=props.public,
            security_groups=[lb_security_group],
            open_listener=props.public,
            certificate=certificate,
        )

        # Remove all default security groups from the load balancer
        fargate_service.load_balancer.connections.security_groups.clear()

        # Add our custom security group
        fargate_service.load_balancer.add_security_group(lb_security_group)

    def _configure_network_acls(self, vpc: ec2.Vpc, whitelist_ips: Optional[List[str]]) -> None:
        """
        Configure Network ACLs to allow traffic on port 6379 for Redis.
        """
        if not whitelist_ips:
            return

        # Get the default Network ACL for the VPC
        default_network_acl = ec2.NetworkAcl.from_network_acl_id(self, "DefaultNetworkAcl", vpc.vpc_default_network_acl)

        # Add inbound rule for Redis
        for ip in whitelist_ips:
            default_network_acl.add_entry(
                id=f"AllowRedisInbound-{ip.replace('.', '-')}",
                cidr=ec2.AclCidr.ipv4(ip),
                rule_number=100,
                traffic=ec2.AclTraffic.tcp_port(6379),
                direction=ec2.TrafficDirection.INGRESS,
                rule_action=ec2.Action.ALLOW,
            )

        # Add outbound rule for Redis
        for ip in whitelist_ips:
            default_network_acl.add_entry(
                id=f"AllowRedisOutbound-{ip.replace('.', '-')}",
                cidr=ec2.AclCidr.ipv4(ip),
                rule_number=100,
                traffic=ec2.AclTraffic.tcp_port(6379),
                direction=ec2.TrafficDirection.EGRESS,
                rule_action=ec2.Action.ALLOW,
            )