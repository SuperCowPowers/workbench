from aws_cdk import (
    Environment,
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_batch as batch,
    Duration,
    Size,
)
from constructs import Construct
from typing import Any, List, Dict
from dataclasses import dataclass, field


@dataclass
class WorkbenchComputeStackProps:
    workbench_bucket: str
    batch_role_arn: str
    existing_vpc_id: str = None
    subnet_ids: List[str] = field(default_factory=list)  # Optional subnets ids for the Batch compute environment


class WorkbenchComputeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        props: WorkbenchComputeStackProps,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench Compute: Defines the compute infrastructure for Workbench including Batch resources."
        super().__init__(scope, construct_id, env=env, description=desc, **kwargs)

        # Print the environment details
        print("Environment")
        print(env)

        # Grab our properties
        self.workbench_bucket = props.workbench_bucket
        self.batch_role_arn = props.batch_role_arn
        self.existing_vpc_id = props.existing_vpc_id
        self.subnet_ids = props.subnet_ids

        # Import the existing batch role
        self.workbench_batch_role = iam.Role.from_role_arn(
            self, "ImportedBatchRole", self.batch_role_arn
        )

        # Batch Compute Environment and Job Queue
        self.batch_compute_environment = self.create_batch_compute_environment()
        self.batch_job_queue = self.create_batch_job_queue()
        self.batch_job_definitions = self.create_batch_job_definitions()

    #####################
    #   Batch Compute   #
    #####################
    def create_batch_compute_environment(self) -> batch.FargateComputeEnvironment:
        """Create the Batch compute environment with Fargate."""

        # Do we have an existing VPC we want to use? Otherwise use a default
        if self.existing_vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "ImportedVPC", vpc_id=self.existing_vpc_id)
        else:
            vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Use specific subnets if provided, otherwise let CDK choose
        vpc_subnets = None
        if self.subnet_ids:
            vpc_subnets = ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(self, f"BatchSubnet{i}", subnet_id)
                    for i, subnet_id in enumerate(self.subnet_ids)
                ]
            )

        return batch.FargateComputeEnvironment(
            self,
            "WorkbenchBatchComputeEnvironment",
            compute_environment_name="workbench-compute-env",
            vpc=vpc,
            vpc_subnets=vpc_subnets,
        )

    def create_batch_job_queue(self) -> batch.JobQueue:
        """Create the Batch job queue."""
        return batch.JobQueue(
            self,
            "WorkbenchBatchJobQueue",
            job_queue_name="workbench-job-queue",
            compute_environments=[
                batch.OrderedComputeEnvironment(compute_environment=self.batch_compute_environment, order=1)
            ],
        )

    def create_batch_job_definitions(self) -> Dict[str, batch.EcsJobDefinition]:
        """Create ML pipeline job definitions in small/medium/large tiers."""

        ecr_image_uri = f"507740646243.dkr.ecr.{self.region}.amazonaws.com/aws-ml-images/py312-ml-pipelines:0.1"
        tiers = {
            "small": (2, 4096),  # 2 vCPU, 4GB RAM
            "medium": (4, 8192),  # 4 vCPU, 8GB RAM
            "large": (8, 16384),  # 8 vCPU, 16GB RAM
        }

        job_definitions = {}
        for size, (cpu, memory_mib) in tiers.items():
            job_definitions[size] = batch.EcsJobDefinition(
                self,
                f"JobDef{size.capitalize()}",
                job_definition_name=f"workbench-batch-{size}",
                container=batch.EcsFargateContainerDefinition(
                    self,
                    f"Container{size.capitalize()}",
                    cpu=cpu,
                    memory=Size.mebibytes(memory_mib),
                    image=ecs.ContainerImage.from_registry(ecr_image_uri),
                    job_role=self.workbench_batch_role,
                    execution_role=self.workbench_batch_role,
                    environment={
                        "WORKBENCH_BUCKET": self.workbench_bucket,
                        "PYTHONUNBUFFERED": "1",
                    },
                ),
                timeout=Duration.hours(3),
            )

        return job_definitions
