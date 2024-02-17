import os
import boto3
import aws_cdk as cdk
from pprint import pprint

from sageworks_dashboard_lite.sageworks_dashboard_stack import SageworksDashboardStack, SageworksDashboardStackProps

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
    sageworks_api_key = cm.get_config("SAGEWORKS_API_KEY")
    existing_vpc_id = cm.get_config("SAGEWORKS_VPC_ID")
    existing_subnet_ids = cm.get_config("SAGEWORKS_SUBNET_IDS")
    config_ips = cm.get_config("SAGEWORKS_WHITELIST", "")
    whitelist_ips = [ip.strip() for ip in config_ips.split(",") if ip.strip()]
    config_prefix = cm.get_config("SAGEWORKS_PREFIX_LISTS", "")
    whitelist_prefix_lists = [ip.strip() for ip in config_prefix.split(",") if ip.strip()]
    certificate_arn = cm.get_config("SAGEWORKS_CERTIFICATE_ARN")
except ImportError:
    print("SageWorks Configuration Manager Not Found...")
    print("Set the SAGEWORKS_CONFiG Env var and run again...")
    raise SystemExit(1)


# TODO: Add in a security group and pass as a prop to the stack

app = cdk.App()
SageworksDashboardStack(
    app,
    "SageworksDashboard",
    env={"account": aws_account, "region": aws_region},
    props=SageworksDashboardStackProps(
        sageworks_bucket=sageworks_bucket,
        sageworks_api_key=sageworks_api_key,
        existing_vpc_id=existing_vpc_id,
        existing_subnet_ids=existing_subnet_ids,
        whitelist_ips=whitelist_ips,
        whitelist_prefix_lists=whitelist_prefix_lists,
        certificate_arn=certificate_arn,
        public=True,
    ),
)

app.synth()
