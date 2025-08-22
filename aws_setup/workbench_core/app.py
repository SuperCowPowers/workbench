import os
import boto3
import logging
import aws_cdk as cdk
from workbench_core.workbench_core_stack import WorkbenchCoreStack, WorkbenchCoreStackProps

# Initialize ConfigManager with error handling
try:
    # Temporarily disable logging
    # logging.disable(logging.CRITICAL)
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
except Exception as e:
    print(f"Workbench ConfigManager initialization failed: {e}")
    print("Falling back to environment variables only")
    cm = None
finally:
    # Re-enable logging
    logging.disable(logging.NOTSET)

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# Get configurations for Workbench Bucket, SSO Group, and Additional Buckets
workbench_bucket = (cm and cm.get_config("WORKBENCH_BUCKET")) or os.getenv("WORKBENCH_BUCKET")
if not workbench_bucket:
    print("Error: WORKBENCH_BUCKET is required but not found in config or environment variables.")
    exit(1)

# SSO Groups (Groups that can assume the Workbench roles)
sso_groups_str = (cm and cm.get_config("WORKBENCH_SSO_GROUPS")) or os.getenv("WORKBENCH_SSO_GROUPS", "")
sso_groups = [group.strip() for group in sso_groups_str.split(",") if group.strip()] if sso_groups_str else []

# Additional Buckets (Extra S3 buckets to grant access to)
additional_buckets_str = (cm and cm.get_config("WORKBENCH_ADDITIONAL_BUCKETS")) or os.getenv(
    "WORKBENCH_ADDITIONAL_BUCKETS", ""
)
additional_buckets = (
    [bucket.strip() for bucket in additional_buckets_str.split(",") if bucket.strip()] if additional_buckets_str else []
)
existing_vpc_id = (cm and cm.get_config("WORKBENCH_VPC_ID")) or os.getenv("WORKBENCH_VPC_ID")
subnet_ids = (cm and cm.get_config("WORKBENCH_SUBNET_IDS")) or os.getenv("WORKBENCH_SUBNET_IDS", "")

# Log the configuration for transparency
print("Configuration:")
print(f"  WORKBENCH_BUCKET: {workbench_bucket}")
print(f"  WORKBENCH_SSO_GROUPS: {sso_groups}")
print(f"  WORKBENCH_ADDITIONAL_BUCKETS: {additional_buckets}")
print(f"  WORKBENCH_VPC_ID: {existing_vpc_id}")
print(f"  WORKBENCH_SUBNET_IDS: {subnet_ids}")

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
        sso_groups=sso_groups,
        additional_buckets=additional_buckets,
        existing_vpc_id=existing_vpc_id,
        subnet_ids=subnet_ids,
    ),
)

app.synth()
