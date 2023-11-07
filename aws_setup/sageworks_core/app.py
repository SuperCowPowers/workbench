import os
import boto3
import aws_cdk as cdk

from sageworks_core.sageworks_core_stack import SageworksCoreStack, SageworksCoreStackProps


# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# We'd like to set up our parameters here
sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
sageworks_role_name = os.environ.get("SAGEWORKS_ROLE", "SageWorks-ExecutionRole")
sso_group = os.environ.get("SAGEWORKS_SSO_GROUP")


# Our CDK App
app = cdk.App()

# Note: We might want to look into this
# env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
env = cdk.Environment(account=aws_account, region=aws_region)

sandbox_stack = SageworksCoreStack(
    app,
    "SageworksCore",
    env=env,
    props=SageworksCoreStackProps(
        sageworks_bucket=sageworks_bucket, sageworks_role_name=sageworks_role_name, sso_group=sso_group
    ),
)

app.synth()
