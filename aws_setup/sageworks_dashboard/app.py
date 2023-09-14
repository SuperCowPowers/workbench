import os
import boto3
import aws_cdk as cdk

from sageworks_dashboard.sageworks_dashboard_stack import SageworksDashboardStack, SageworksDashboardStackProps

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client('sts').get_caller_identity().get('Account')
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# We'd like to set up our parameters here
sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
existing_vpc_id = os.environ.get("SAGEWORKS_VPC_ID")
existing_subnet_ids = os.environ.get("SAGEWORKS_SUBNET_IDS")
whitelist_ips = [ip.strip() for ip in os.environ.get("SAGEWORKS_WHITELIST", "").split(",") if ip.strip()]

app = cdk.App()
SageworksDashboardStack(
    app,
    "SageworksDashboard",
    env={
        'account': aws_account,
        'region': aws_region
    },
    props=SageworksDashboardStackProps(
        sageworks_bucket=sageworks_bucket,
        existing_vpc_id=existing_vpc_id,
        existing_subnet_ids=existing_subnet_ids,
        whitelist_ips=whitelist_ips,
    )
)

app.synth()
