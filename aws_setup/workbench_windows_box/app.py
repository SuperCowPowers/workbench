import boto3
import aws_cdk as cdk

from workbench_windows_box.workbench_windows_box_stack import WorkbenchWindowsBoxStack

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

app = cdk.App()
env = cdk.Environment(account=aws_account, region=aws_region)

WorkbenchWindowsBoxStack(app, "WorkbenchWindowsBox", env=env)

app.synth()
