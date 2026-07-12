"""On-demand SSM port-forwarding tunnel to the Workbench Redis (ElastiCache) cluster.

The Workbench Redis cluster lives in private subnets and is not reachable from a
laptop directly. This script brings up (or starts) a tiny, **keyless** EC2
instance in a private subnet of the Workbench VPC, registers it with AWS Systems
Manager (SSM), and opens a port-forwarding tunnel from a local port to the private
Redis endpoint — **no SSH key, no public IP, no inbound security-group rule**.

The tunnel runs in the foreground and reopens the session automatically if it drops
(e.g. the SSM idle timeout); **press Ctrl-C to close it, and the instance is stopped
automatically** (no standing cost, nothing left running). Run it again to reconnect
(it restarts the same instance).

Everything is discovered, not hardcoded: the VPC and private subnets come from the
active Workbench config (``WORKBENCH_VPC_ID`` / ``WORKBENCH_SUBNET_IDS``) and the
Redis endpoint is looked up from ElastiCache. Run with AWS credentials that can
manage EC2/IAM/SSM (the same admin profile used for ``cdk deploy``).

Prereq: the AWS CLI ``session-manager-plugin`` (``brew install session-manager-plugin``).

Usage:
    WORKBENCH_CONFIG=.../admin.json redis_tunnel              # open tunnel (Ctrl-C closes + stops)
    WORKBENCH_CONFIG=.../admin.json redis_tunnel --terminate  # remove the instance + its SG
"""

import argparse
import shutil
import subprocess
import time

import boto3
from botocore.exceptions import ClientError

from workbench.utils.config_manager import ConfigManager

# Generic resource names — nothing account-specific lives in this file.
TAG_NAME = "workbench-redis-tunnel"
ROLE_NAME = "workbench-ssm-tunnel-role"
PROFILE_NAME = "workbench-ssm-tunnel-profile"
SG_NAME = "workbench-redis-tunnel-sg"
REDIS_CLUSTER_MATCH = "workbenchredis"  # ElastiCache lower-cases the dashboard stack's "WorkbenchRedis"
INSTANCE_TYPE = "t3.micro"
DEFAULT_LOCAL_PORT = 6380
REDIS_PORT = 6379
SSM_MANAGED_POLICY = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
AL2023_AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
ASSUME_ROLE_DOC = (
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
    '"Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
)


def require_plugin() -> None:
    """Fail fast (before provisioning anything) if the SSM session-manager-plugin is missing."""
    if shutil.which("session-manager-plugin") is None:
        raise SystemExit(
            "The AWS 'session-manager-plugin' is not installed.\n"
            "  macOS:  brew install session-manager-plugin\n"
            "  Other:  https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html"  # noqa: E501
        )


def ensure_instance_profile(iam) -> str:
    """Create (idempotently) the IAM role + instance profile that lets the instance register with SSM."""
    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
        return PROFILE_NAME
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"Creating IAM instance profile '{PROFILE_NAME}' …")
    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=ASSUME_ROLE_DOC,
            Description="Workbench SSM Redis tunnel instance role",
        )
    except iam.exceptions.EntityAlreadyExistsException:
        # Role already exists — idempotent
        pass
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_MANAGED_POLICY)
    try:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
    except iam.exceptions.EntityAlreadyExistsException:
        # Profile already exists — idempotent
        pass
    return PROFILE_NAME


def ensure_security_group(ec2, vpc_id: str) -> str:
    """Find or create an egress-only SG for the tunnel instance (the VPC default SG has its rules stripped)."""
    found = ec2.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [SG_NAME]}]
    )["SecurityGroups"]
    if found:
        return found[0]["GroupId"]
    sg = ec2.create_security_group(
        GroupName=SG_NAME, Description="Workbench SSM Redis tunnel (egress only)", VpcId=vpc_id
    )
    # New SGs come with allow-all egress and no ingress — exactly what we want.
    return sg["GroupId"]


def discover_redis_endpoint(elasticache) -> str:
    """Look up the Workbench Redis endpoint from ElastiCache (not hardcoded)."""
    clusters = elasticache.describe_cache_clusters(ShowCacheNodeInfo=True)["CacheClusters"]
    matches = [c for c in clusters if REDIS_CLUSTER_MATCH in c.get("CacheClusterId", "").lower()]
    if not matches:
        raise SystemExit("Could not find the Workbench Redis cluster in ElastiCache.")
    if len(matches) > 1:
        print(f"Warning: multiple Redis clusters matched; using '{matches[0]['CacheClusterId']}'")
    return matches[0]["CacheNodes"][0]["Endpoint"]["Address"]


def find_instance(ec2):
    """Return the existing tunnel instance (any non-terminated state), or None."""
    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )["Reservations"]
    for r in reservations:
        for inst in r["Instances"]:
            return inst
    return None


def ensure_running(ec2, ssm, subnet_id: str, profile_name: str, sg_id: str) -> str:
    """Create the tunnel instance if missing, start it if stopped, and return its id (running)."""
    inst = find_instance(ec2)
    if inst:
        iid, state = inst["InstanceId"], inst["State"]["Name"]
        if state != "running":
            print(f"Starting tunnel instance {iid} (was {state}) …")
            ec2.start_instances(InstanceIds=[iid])
            ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        else:
            print(f"Tunnel instance {iid} already running")
        return iid

    ami = ssm.get_parameter(Name=AL2023_AMI_PARAM)["Parameter"]["Value"]
    print(f"Launching tunnel instance ({INSTANCE_TYPE}, AL2023 {ami}) in {subnet_id} …")
    for attempt in range(12):  # retry while the new instance profile propagates
        try:
            run = ec2.run_instances(
                ImageId=ami,
                InstanceType=INSTANCE_TYPE,
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet_id,
                SecurityGroupIds=[sg_id],
                IamInstanceProfile={"Name": profile_name},
                MetadataOptions={"HttpTokens": "required"},  # IMDSv2 only
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": TAG_NAME}]}],
            )
            break
        except ClientError as e:
            if "Invalid IAM Instance Profile" in str(e) and attempt < 11:
                time.sleep(5)
                continue
            raise
    iid = run["Instances"][0]["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    return iid


def wait_ssm_registered(ssm, instance_id: str, timeout: int = 180) -> None:
    """Wait until the SSM agent on the instance has registered (needed before start-session works)."""
    print("Waiting for the SSM agent to register", end="", flush=True)
    for _ in range(timeout // 5):
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [instance_id]}])
        if info["InstanceInformationList"]:
            print(" — registered.")
            return
        print(".", end="", flush=True)
        time.sleep(5)
    print(f"\nTimed out after {timeout}s (the agent may still come up — re-run shortly).")


def open_tunnel(instance_id: str, endpoint: str, local_port: int, region: str) -> None:
    """Run `aws ssm start-session` port-forwarding in the foreground, reconnecting whenever
    the session drops (SSM idle timeout, network blip). Blocks until Ctrl-C."""
    cmd = [
        "aws",
        "ssm",
        "start-session",
        "--target",
        instance_id,
        "--document-name",
        "AWS-StartPortForwardingSessionToRemoteHost",
        "--parameters",
        f"host={endpoint},portNumber={REDIS_PORT},localPortNumber={local_port}",
    ]
    if region:
        cmd += ["--region", region]

    print("\n" + "=" * 70)
    print(f"Tunnel open:  localhost:{local_port}  →  {endpoint}:{REDIS_PORT}")
    print(f"In another terminal:  redis-cli -h localhost -p {local_port}")
    print("Sessions that drop (SSM idle timeout, etc.) are reopened automatically.")
    print("Press Ctrl-C here to close the tunnel (the instance is stopped automatically).")
    print("=" * 70 + "\n")

    rapid_failures = 0
    while True:
        started = time.monotonic()
        subprocess.run(cmd)
        # Session ended on its own (Ctrl-C raises KeyboardInterrupt out of run()).
        if time.monotonic() - started < 30:
            rapid_failures += 1
            if rapid_failures >= 3:
                print("Session keeps failing immediately — giving up.")
                return
        else:
            rapid_failures = 0
        print("\nSession dropped — reconnecting …\n")
        time.sleep(2)


def terminate(ec2, instance_id: str, vpc_id: str) -> None:
    print(f"Terminating {instance_id} …")
    ec2.terminate_instances(InstanceIds=[instance_id])
    ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
    for sg in ec2.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [SG_NAME]}]
    )["SecurityGroups"]:
        try:
            ec2.delete_security_group(GroupId=sg["GroupId"])
            print(f"Deleted SG {sg['GroupId']}")
        except ClientError as e:
            print(f"Could not delete SG {sg['GroupId']}: {e}")
    print("Done — no tunnel resources left behind.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--terminate", action="store_true", help="Terminate the tunnel instance and delete its SG, then exit"
    )
    parser.add_argument(
        "--local-port", type=int, default=DEFAULT_LOCAL_PORT, help="Local port to forward (default 6380)"
    )
    args = parser.parse_args()

    cm = ConfigManager()
    vpc_id = cm.get_config("WORKBENCH_VPC_ID")
    subnet_ids = cm.get_config("WORKBENCH_SUBNET_IDS") or []
    if not vpc_id or not subnet_ids:
        raise SystemExit("WORKBENCH_VPC_ID / WORKBENCH_SUBNET_IDS are not set in the active Workbench config.")

    session = boto3.Session()
    region = session.region_name
    ec2 = session.client("ec2")

    if args.terminate:
        inst = find_instance(ec2)
        if not inst:
            print("No tunnel instance found.")
            return
        terminate(ec2, inst["InstanceId"], vpc_id)
        return

    # Open a tunnel: bring the instance up, forward the port, and always stop on exit.
    require_plugin()
    iam, ssm, elasticache = session.client("iam"), session.client("ssm"), session.client("elasticache")
    profile_name = ensure_instance_profile(iam)
    sg_id = ensure_security_group(ec2, vpc_id)
    endpoint = discover_redis_endpoint(elasticache)
    instance_id = ensure_running(ec2, ssm, subnet_ids[0], profile_name, sg_id)
    wait_ssm_registered(ssm, instance_id)

    try:
        open_tunnel(instance_id, endpoint, args.local_port, region)
    except KeyboardInterrupt:
        # Ctrl-C: fall through to instance cleanup
        pass
    finally:
        print(f"\nStopping tunnel instance {instance_id} …")
        ec2.stop_instances(InstanceIds=[instance_id])
        print("Stopped. Run again to reconnect, or --terminate to remove it entirely.")


if __name__ == "__main__":
    main()
