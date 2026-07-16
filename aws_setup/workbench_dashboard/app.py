import boto3
import aws_cdk as cdk
from pprint import pprint

from workbench_dashboard.workbench_dashboard_stack import WorkbenchDashboardStack, WorkbenchDashboardStackProps

# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# When you want a different docker image change this line
dashboard_image = "public.ecr.aws/m6i5k1r2/workbench_dashboard:v0_8_408_amd64"

# Workbench Configuration
try:
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
    pprint(cm.config)
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
    workbench_api_key = cm.get_config("WORKBENCH_API_KEY")
    workbench_plugins = cm.get_config("WORKBENCH_PLUGINS")
    workbench_themes = cm.get_config("WORKBENCH_THEMES")
    # Resolves to the config value, or falls back to s3://<WORKBENCH_BUCKET>/ml_pipelines
    ml_pipelines_root = cm.get_config("ML_PIPELINES_ROOT")
    existing_vpc_id = cm.get_config("WORKBENCH_VPC_ID")
    existing_subnet_ids = cm.get_config("WORKBENCH_SUBNET_IDS")
    config_ips = cm.get_config("WORKBENCH_WHITELIST", "")
    whitelist_ips = [ip.strip() for ip in config_ips.split(",") if ip.strip()]
    config_prefix = cm.get_config("WORKBENCH_PREFIX_LISTS", "")
    whitelist_prefix_lists = [ip.strip() for ip in config_prefix.split(",") if ip.strip()]
    certificate_arn = cm.get_config("WORKBENCH_CERTIFICATE_ARN")
    public = str(cm.get_config("WORKBENCH_DASHBOARD_PUBLIC", "false")).strip().lower() in ("1", "true", "yes", "on")
    desired_count = int(cm.get_config("WORKBENCH_DASHBOARD_TASK_COUNT", 1))
except ImportError:
    print("Workbench Configuration Manager Not Found...")
    print("Set the WORKBENCH_CONFiG Env var and run again...")
    raise SystemExit(1)


# TODO: Add in a security group and pass as a prop to the stack

app = cdk.App()
WorkbenchDashboardStack(
    app,
    "WorkbenchDashboard",
    env={"account": aws_account, "region": aws_region},
    props=WorkbenchDashboardStackProps(
        dashboard_image=dashboard_image,
        workbench_bucket=workbench_bucket,
        workbench_api_key=workbench_api_key,
        workbench_plugins=workbench_plugins,
        workbench_themes=workbench_themes,
        ml_pipelines_root=ml_pipelines_root,
        existing_vpc_id=existing_vpc_id,
        existing_subnet_ids=existing_subnet_ids,
        whitelist_ips=whitelist_ips,
        whitelist_prefix_lists=whitelist_prefix_lists,
        certificate_arn=certificate_arn,
        public=public,
        desired_count=desired_count,
    ),
)

app.synth()
