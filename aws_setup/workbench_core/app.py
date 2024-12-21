import os
import boto3
import aws_cdk as cdk
from workbench_core.workbench_core_stack import WorkbenchCoreStack, WorkbenchCoreStackProps

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# Read required configuration from environment variables
workbench_bucket = os.getenv("WORKBENCH_BUCKET")
if not workbench_bucket:
    print("Error: The environment variable WORKBENCH_BUCKET is required but not set.")
    exit(1)

# Read optional configurations with defaults
workbench_role_name = os.getenv("WORKBENCH_ROLE", "Workbench-ExecutionRole")
sso_group = os.getenv("WORKBENCH_SSO_GROUP")  # Optional, default to None if not set
additional_buckets = [
    bucket.strip() for bucket in os.getenv("WORKBENCH_ADDITIONAL_BUCKETS", "").split(",") if bucket.strip()
]

# Log the configuration for transparency
print("Configuration:")
print(f"  WORKBENCH_BUCKET: {workbench_bucket}")
print(f"  WORKBENCH_ROLE: {workbench_role_name}")
print(f"  WORKBENCH_SSO_GROUP: {sso_group}")
print(f"  WORKBENCH_ADDITIONAL_BUCKETS: {additional_buckets}")

# Our CDK App and Environment
app = cdk.App()
env = cdk.Environment(account=aws_account, region=aws_region)

# Create the Workbench Core Stack
sandbox_stack = WorkbenchCoreStack(
    app,
    "WorkbenchCore",
    env=env,
    props=WorkbenchCoreStackProps(
        workbench_bucket=workbench_bucket,
        workbench_role_name=workbench_role_name,
        sso_group=sso_group,
        additional_buckets=additional_buckets,
    ),
)

app.synth()
