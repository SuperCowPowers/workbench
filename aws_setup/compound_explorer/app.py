import boto3
import aws_cdk as cdk
from pprint import pprint

from compound_explorer.compound_explorer_stack import CompoundExplorerStack, CompoundExplorerStackProps

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# When you want a different docker image change this line
dashboard_image = "public.ecr.aws/m6i5k1r2/compound_explorer:v0_8_90_amd64"

# Workbench Configuration (with some Explorer parameters)
try:
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
    pprint(cm.config)
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
    workbench_api_key = cm.get_config("WORKBENCH_API_KEY")
    workbench_plugins = cm.get_config("WORKBENCH_PLUGINS")
    certificate_arn = cm.get_config("EXPLORER_CERTIFICATE_ARN")
except ImportError:
    print("Workbench Configuration Manager Not Found...")
    print("Set the WORKBENCH_CONFiG Env var and run again...")
    raise SystemExit(1)


app = cdk.App()
CompoundExplorerStack(
    app,
    "CompoundExplorer",
    env={"account": aws_account, "region": aws_region},
    props=CompoundExplorerStackProps(
        dashboard_image=dashboard_image,
        workbench_bucket=workbench_bucket,
        workbench_api_key=workbench_api_key,
        workbench_plugins=workbench_plugins,
        certificate_arn=certificate_arn,
        public=True,
    ),
)

app.synth()
