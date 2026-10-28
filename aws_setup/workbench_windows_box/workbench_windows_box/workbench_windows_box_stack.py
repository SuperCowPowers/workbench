from aws_cdk import (
    Environment,
    Stack,
    CfnOutput,
    Duration,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
)
from constructs import Construct
from typing import Any

WINDOWS_AMI_PARAMETER = "/aws/service/ami-windows-latest/Windows_Server-2025-English-Full-Base"
PYTHON_VERSION = "3.13.17"

# First-boot setup: python.org Python (all users, on PATH) and the AWS CLI. Long paths
# stay at the Windows default (off), matching a typical user machine.
SETUP_COMMANDS = f"""
$ProgressPreference = "SilentlyContinue"
$exe = "python-{PYTHON_VERSION}-amd64.exe"
$installer = "$env:TEMP\\$exe"
Invoke-WebRequest "https://www.python.org/ftp/python/{PYTHON_VERSION}/$exe" -OutFile $installer
Start-Process $installer -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
Start-Process msiexec.exe -ArgumentList "/i https://awscli.amazonaws.com/AWSCLIV2.msi /qn" -Wait
"""


class WorkbenchWindowsBoxStack(Stack):
    """A stop/start Windows Server 2025 instance for hands-on Workbench testing.

    The instance sits in the account's default VPC with no inbound rules: RDP runs
    through an SSM port forward. The AMI is cached in ``cdk.context.json`` so a
    later deploy doesn't replace the instance (and its setup) with a newer image.

    A CloudWatch alarm stops the instance after ``idle_hours`` of low CPU, so a
    forgotten session costs hours, not a month.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        *,
        instance_type: str = "t3.large",
        volume_gb: int = 60,
        idle_hours: int = 2,
        **kwargs: Any,
    ) -> None:
        desc = "Workbench Windows Box: a stop/start Windows Server 2025 instance for hands-on testing."
        super().__init__(scope, construct_id, env=env, description=desc, **kwargs)

        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)
        security_group = ec2.SecurityGroup(
            self,
            "SecurityGroup",
            vpc=vpc,
            description="Workbench Windows Box: no inbound, RDP goes through an SSM port forward",
            allow_all_outbound=True,
        )
        role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )
        # The private key lands in SSM Parameter Store; it decrypts the Administrator password
        key_pair = ec2.KeyPair(self, "KeyPair", key_pair_name="workbench-windows-box")

        user_data = ec2.UserData.for_windows()
        user_data.add_commands(SETUP_COMMANDS)

        self.instance = ec2.Instance(
            self,
            "Instance",
            instance_name="workbench-windows-box",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ec2.MachineImage.from_ssm_parameter(
                WINDOWS_AMI_PARAMETER, os=ec2.OperatingSystemType.WINDOWS, cached_in_context=True
            ),
            security_group=security_group,
            role=role,
            key_pair=key_pair,
            user_data=user_data,
            require_imdsv2=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_gb, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True
                    ),
                )
            ],
        )

        idle_alarm = cloudwatch.Alarm(
            self,
            "IdleStopAlarm",
            alarm_description=f"Stop the Workbench Windows Box after {idle_hours} idle hours",
            metric=cloudwatch.Metric(
                namespace="AWS/EC2",
                metric_name="CPUUtilization",
                dimensions_map={"InstanceId": self.instance.instance_id},
                statistic="Average",
                period=Duration.minutes(15),
            ),
            threshold=5,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=idle_hours * 4,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        idle_alarm.add_alarm_action(cw_actions.Ec2Action(cw_actions.Ec2InstanceAction.STOP))

        CfnOutput(self, "InstanceId", value=self.instance.instance_id, description="Start/stop and port-forward target")
        CfnOutput(
            self,
            "KeyParameter",
            value=f"/ec2/keypair/{key_pair.key_pair_id}",
            description="SSM parameter holding the private key that decrypts the Administrator password",
        )
