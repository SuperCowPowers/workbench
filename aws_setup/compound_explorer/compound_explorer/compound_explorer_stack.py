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
        certificate_arn: Optional[str] = None,
        public: bool = False,
    ):
        self.dashboard_image = dashboard_image
        self.workbench_bucket = workbench_bucket
        self.workbench_api_key = workbench_api_key
        self.workbench_plugins = workbench_plugins
        self.certificate_arn = certificate_arn
        self.public = public


class CompoundExplorerStack(Stack):
    def __init__(self, scope: Construct, id: str, props: CompoundExplorerStackProps, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        if not props.workbench_bucket:
            raise ValueError("workbench_bucket is a required property")

        # Create a ECS cluster using a NEW VPC
        cluster = ecs.Cluster(self, "ExplorerCluster", vpc=ec2.Vpc(self, "ExplorerVpc", max_azs=2))

        # Import the existing Workbench-ExecutionRole
        workbench_execution_role = iam.Role.from_role_arn(
            self, "ImportedWorkbenchExecutionRole", f"arn:aws:iam::{self.account}:role/Workbench-ExecutionRole"
        )

        # Setup CloudWatch logs
        log_group = logs.LogGroup(self, "ExplorerLogGroup")

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
                "WORKBENCH_BUCKET": props.workbench_bucket,
                "WORKBENCH_API_KEY": props.workbench_api_key,
                "WORKBENCH_PLUGINS": props.workbench_plugins,
            },
            logging=ecs.LogDriver.aws_logs(stream_prefix="CompoundExplorer", log_group=log_group),
        )
        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # Create a NEW Security Group for the Load Balancer
        lb_security_group = ec2.SecurityGroup(self, "LoadBalancerSecurityGroup", vpc=cluster.vpc)

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
