import boto3
import aws_cdk as cdk
from aws_cdk import Fn
from workbench_compute.workbench_compute_stack import WorkbenchComputeStack, WorkbenchComputeStackProps

# Initialize Workbench ConfigManager
try:
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
except ImportError:
    print("Workbench Configuration Manager Not Found...")
    print("Set the WORKBENCH_CONFIG Env var and run again...")
    raise SystemExit(1)

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# Get configurations needed for compute stack
workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
if not workbench_bucket:
    print("Error: WORKBENCH_BUCKET is required but not found in config.")
    exit(1)


# Get the Batch and Lambda Roles from WorkbenchCore stack outputs
batch_role_arn = Fn.import_value("WorkbenchCore-BatchRoleArn")
lambda_role_arn = Fn.import_value("WorkbenchCore-LambdaRoleArn")

# VPC and subnet configuration (optional)
existing_vpc_id = cm.get_config("WORKBENCH_VPC_ID")
subnet_ids = cm.get_config("WORKBENCH_SUBNET_IDS") or []

# Log the configuration for transparency
print("Configuration:")
print(f"  WORKBENCH_BUCKET: {workbench_bucket}")
print(f"  WORKBENCH_BATCH_ROLE_ARN: {batch_role_arn}")
print(f"  WORKBENCH_LAMBDA_ROLE_ARN: {lambda_role_arn}")
print(f"  WORKBENCH_VPC_ID: {existing_vpc_id}")
print(f"  WORKBENCH_SUBNET_IDS: {subnet_ids}")

# Our CDK App and Environment
app = cdk.App()
env = cdk.Environment(account=aws_account, region=aws_region)

# Create the Workbench Compute Stack
compute_stack = WorkbenchComputeStack(
    app,
    "WorkbenchCompute",
    env=env,
    props=WorkbenchComputeStackProps(
        workbench_bucket=workbench_bucket,
        batch_role_arn=batch_role_arn,
        lambda_role_arn=lambda_role_arn,
        existing_vpc_id=existing_vpc_id,
        subnet_ids=subnet_ids,
    ),
)

app.synth()
