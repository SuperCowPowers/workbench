"""Delete orphaned AWS Glue network interfaces (ENIs).

Glue jobs with a VPC connection create one ENI per worker in the connection's subnet and
delete them at job end using the job's IAM role. When that role can't delete them (missing
``ec2:DeleteNetworkInterface``, or the role itself was deleted), the ENIs are left detached
("available") and each one keeps its private IP, eventually exhausting the subnet.

Targets ENIs that are ``available`` and described as "Attached to Glue using role: ...".
Uses the caller's AWS_PROFILE directly (not the Workbench role), so run it with an admin
profile once per AWS environment.

Usage:
    python scripts/admin/delete_orphaned_glue_enis.py            # dry-run (default)
    python scripts/admin/delete_orphaned_glue_enis.py --apply    # delete the ENIs
"""

import argparse
import sys
from collections import Counter

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

GLUE_DESCRIPTION = "Attached to Glue using role: *"


def orphaned_glue_enis(ec2) -> list[dict]:
    """All detached Glue ENIs in the region."""
    filters = [
        {"Name": "status", "Values": ["available"]},
        {"Name": "description", "Values": [GLUE_DESCRIPTION]},
    ]
    enis = []
    for page in ec2.get_paginator("describe_network_interfaces").paginate(Filters=filters):
        enis.extend(page["NetworkInterfaces"])
    return enis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Delete the ENIs (default is a dry-run)")
    parser.add_argument("--region", help="AWS region (default: the profile's region)")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    ec2 = session.client("ec2", config=Config(retries={"mode": "adaptive", "max_attempts": 10}))

    enis = orphaned_glue_enis(ec2)
    print(f"Account: {account} | Region: {session.region_name} | Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Orphaned Glue ENIs: {len(enis)}\n")
    if not enis:
        return 0

    by_subnet = Counter((e["SubnetId"], e["AvailabilityZone"]) for e in enis)
    by_role = Counter(e["Description"].rsplit("/", 1)[-1] for e in enis)
    print("By subnet:")
    for (subnet, az), n in by_subnet.most_common():
        print(f"  {n:6d}  {subnet} ({az})")
    print("By Glue job role:")
    for role, n in by_role.most_common():
        print(f"  {n:6d}  {role}")

    if not args.apply:
        print("\nDry-run: pass --apply to delete.")
        return 0

    deleted = failed = 0
    for i, eni in enumerate(enis, 1):
        try:
            ec2.delete_network_interface(NetworkInterfaceId=eni["NetworkInterfaceId"])
            deleted += 1
        except ClientError as e:
            # Attached since the listing (Glue reused it) or already gone
            print(f"  ! {eni['NetworkInterfaceId']}: {e.response['Error']['Code']}")
            failed += 1
        if i % 250 == 0:
            print(f"  ... {i}/{len(enis)}")

    print(f"\nDeleted: {deleted} | Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
