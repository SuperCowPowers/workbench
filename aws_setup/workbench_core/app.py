import os
import boto3
import aws_cdk as cdk
from workbench_core.workbench_core_stack import WorkbenchCoreStack, WorkbenchCoreStackProps

# Initialize ConfigManager with error handling
try:
    from workbench.utils.log_utils import silence_logs
    with silence_logs():
        from workbench.utils.config_manager import ConfigManager
        cm = ConfigManager()
except Exception as e:
    print(f"Workbench ConfigManager initialization failed: {e}")
    print("Falling back to environment variables only")
    cm = None

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# Get configurations with fallback
workbench_bucket = (cm and cm.get_config("WORKBENCH_BUCKET")) or os.getenv("WORKBENCH_BUCKET")
if not workbench_bucket:
    print("Error: WORKBENCH_BUCKET is required but not found in config or environment variables.")
    exit(1)

sso_group = (cm and cm.get_config("WORKBENCH_SSO_GROUP")) or os.getenv("WORKBENCH_SSO_GROUP")

additional_buckets_str = (cm and cm.get_config("WORKBENCH_ADDITIONAL_BUCKETS")) or os.getenv("WORKBENCH_ADDITIONAL_BUCKETS", "")
additional_buckets = [
    bucket.strip() for bucket in additional_buckets_str.split(",") if bucket.strip()
] if additional_buckets_str else []

# Log the configuration for transparency
print("Configuration:")
print(f"  WORKBENCH_BUCKET: {workbench_bucket}")
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
        sso_group=sso_group,
        additional_buckets=additional_buckets,
    ),
)

app.synth()
