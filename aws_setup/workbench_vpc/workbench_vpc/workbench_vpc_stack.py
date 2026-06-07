from aws_cdk import (
    Environment,
    Stack,
    CfnOutput,
    aws_ec2 as ec2,
)
from constructs import Construct
from typing import Any


class WorkbenchVpcStack(Stack):
    """A small, cost-optimized VPC for Workbench Batch compute.

    Creates one VPC spanning ``max_azs`` Availability Zones, each with a public
    and a private subnet, an Internet Gateway, and ``nat_gateways`` NAT gateways
    (default 1 — a single NAT serves every private subnet, so two-AZ resilience
    costs nothing extra). The CDK ``ec2.Vpc`` construct wires the subnets, route
    tables, IGW, and NAT automatically.

    Batch Fargate runs in the **private** subnets and reaches ECR through the NAT.
    The VPC id and private subnet ids are exported as stack outputs to copy into
    the Workbench config (``WORKBENCH_VPC_ID`` / ``WORKBENCH_SUBNET_IDS``).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        *,
        cidr: str = "10.0.0.0/16",
        max_azs: int = 2,
        nat_gateways: int = 1,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench VPC: A cost-optimized VPC (public/private subnets + NAT) for Workbench Batch compute."
        super().__init__(scope, construct_id, env=env, description=desc, **kwargs)

        self.vpc = ec2.Vpc(
            self,
            "WorkbenchVpc",
            vpc_name="workbench-vpc",
            ip_addresses=ec2.IpAddresses.cidr(cidr),
            max_azs=max_azs,
            nat_gateways=nat_gateways,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24),
            ],
        )

        private_subnet_ids = [subnet.subnet_id for subnet in self.vpc.private_subnets]

        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            export_name="WorkbenchVpc-VpcId",
            description='Set this as "WORKBENCH_VPC_ID" in your Workbench config',
        )
        CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join(private_subnet_ids),
            export_name="WorkbenchVpc-PrivateSubnetIds",
            description='Set these as a JSON array for "WORKBENCH_SUBNET_IDS" in your Workbench config',
        )
