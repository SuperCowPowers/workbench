import boto3
import aws_cdk as cdk
from pprint import pprint

from sageworks_core.sageworks_core_stack import SageworksCoreStack, SageworksCoreStackProps


# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# SageWorks Configuration
try:
    from sageworks.utils.config_manager import ConfigManager

    cm = ConfigManager()
    pprint(cm.config)
    sageworks_bucket = cm.get_config("SAGEWORKS_BUCKET")
    sageworks_role_name = cm.get_config("SAGEWORKS_ROLE")
    sso_group = cm.get_config("SAGEWORKS_SSO_GROUP")
    additional_buckets = cm.get_config("SAGEWORKS_ADDITIONAL_BUCKETS")
except ImportError:
    print("SageWorks Configuration Manager Not Found...")
    print("Set the SAGEWORKS_CONFiG Env var and run again...")
    raise SystemExit(1)


# Our CDK App and Environment
app = cdk.App()
env = cdk.Environment(account=aws_account, region=aws_region)

sandbox_stack = SageworksCoreStack(
    app,
    "SageworksCore",
    env=env,
    props=SageworksCoreStackProps(
        sageworks_bucket=sageworks_bucket,
        sageworks_role_name=sageworks_role_name,
        sso_group=sso_group,
        additional_buckets=additional_buckets.split(",") if additional_buckets else [],
    ),
)

app.synth()
