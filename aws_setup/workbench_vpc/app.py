import boto3
import aws_cdk as cdk

from workbench_vpc.workbench_vpc_stack import WorkbenchVpcStack

# Workbench ConfigManager is optional here — the VPC stack needs no Workbench
# config, but we honor a few overrides if a config is present.
try:
    from workbench.utils.config_manager import ConfigManager

    cm = ConfigManager()
except Exception:
    cm = None


def _cfg(key: str, default):
    if cm is not None:
        val = cm.get_config(key)
        if val not in (None, ""):
            return val
    return default


# Grab the account and region using boto3
session = boto3.session.Session()
aws_account = session.client("sts").get_caller_identity().get("Account")
aws_region = session.region_name
print(f"Account: {aws_account}")
print(f"Region: {aws_region}")

# Optional overrides (sensible defaults match docs/aws_setup/compute_stack.md)
cidr = _cfg("WORKBENCH_VPC_CIDR", "10.0.0.0/16")
max_azs = int(_cfg("WORKBENCH_VPC_MAX_AZS", 2))
nat_gateways = int(_cfg("WORKBENCH_VPC_NAT_GATEWAYS", 1))

print("Configuration:")
print(f"  CIDR: {cidr}")
print(f"  Max AZs: {max_azs}")
print(f"  NAT Gateways: {nat_gateways}")

app = cdk.App()
env = cdk.Environment(account=aws_account, region=aws_region)

WorkbenchVpcStack(
    app,
    "WorkbenchVpc",
    env=env,
    cidr=cidr,
    max_azs=max_azs,
    nat_gateways=nat_gateways,
)

app.synth()
