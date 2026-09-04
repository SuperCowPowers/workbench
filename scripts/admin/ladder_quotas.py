"""Check (and raise) the SageMaker training quotas an account needs to run our training jobs.

Every instance in INSTANCE_LADDERS needs quota behind it. A rung at zero is not a fallback,
it is a guaranteed failure: the ladder waits out CAPACITY_WAIT_SECONDS on a rung the account
can never launch, then drops to the next. The final rung of a ladder matters most, since a
job that reaches it queues there for as long as it takes.

Targets are per ladder because the workloads fan out differently: model builds run many at
once, while a parallel HPO search takes a whole multi-GPU box and rarely runs concurrently.

Exits nonzero when any rung is below target, so it can gate a new account's setup.

Usage:
    AWS_PROFILE=<admin> python scripts/admin/ladder_quotas.py           # report
    AWS_PROFILE=<admin> python scripts/admin/ladder_quotas.py --apply   # request increases
"""

import argparse
import sys

import boto3

from workbench.core.transforms.features_to_model.features_to_model import INSTANCE_LADDERS

SERVICE_CODE = "sagemaker"
QUOTA_SUFFIX = " for training job usage"
OPEN_STATUSES = {"PENDING", "CASE_OPENED"}

# Concurrent instances of that ladder's rungs the account should be able to launch.
LADDER_TARGETS = {
    "gpu": 30,  # single-card chemprop/pytorch builds, the real fan-out
    "cpu": 30,  # XGBoost builds, same fan-out
    "cpu_hpo": 4,  # a search is one box; a handful of concurrent searches
    "gpu_hpo": 4,  # 4 GPUs per box, so a handful of concurrent searches
}


def quota_codes(client) -> dict:
    """Map every "<instance> for training job usage" quota name to its code and default value."""
    codes = {}
    for page in client.get_paginator("list_aws_default_service_quotas").paginate(ServiceCode=SERVICE_CODE):
        for quota in page["Quotas"]:
            if quota["QuotaName"].endswith(QUOTA_SUFFIX):
                codes[quota["QuotaName"]] = (quota["QuotaCode"], quota["Value"])
    return codes


def applied_value(client, code: str, default: float) -> float:
    """The account's quota, which falls back to the AWS default when it was never changed."""
    try:
        return client.get_service_quota(ServiceCode=SERVICE_CODE, QuotaCode=code)["Quota"]["Value"]
    except client.exceptions.NoSuchResourceException:
        return default


def open_request(client, code: str) -> bool:
    """Whether an increase for this quota is already in flight."""
    history = client.list_requested_service_quota_change_history_by_quota(ServiceCode=SERVICE_CODE, QuotaCode=code)[
        "RequestedQuotas"
    ]
    return any(request["Status"] in OPEN_STATUSES for request in history)


def check_rung(client, codes: dict, instance: str, target: int, apply: bool) -> bool:
    """Report one rung, requesting an increase under --apply. True when it meets the target."""
    quota_name = f"{instance}{QUOTA_SUFFIX}"
    if quota_name not in codes:
        print(f"    {instance:<18} {'':>6}  no such SageMaker training quota")
        return False

    code, default = codes[quota_name]
    value = applied_value(client, code, default)
    if value >= target:
        # Service Quotas has no way to lower a quota, so above target is just fine.
        print(f"    {instance:<18} {value:>6.0f}  ok (target {target})")
        return True

    if open_request(client, code):
        print(f"    {instance:<18} {value:>6.0f}  LOW, increase to {target} already requested")
    elif not apply:
        print(f"    {instance:<18} {value:>6.0f}  LOW, would request {target} (--apply)")
    else:
        request = client.request_service_quota_increase(ServiceCode=SERVICE_CODE, QuotaCode=code, DesiredValue=target)[
            "RequestedQuota"
        ]
        print(f"    {instance:<18} {value:>6.0f}  requested {target}  case {request['Id']}")
    return False


def main(apply: bool) -> int:
    session = boto3.Session()
    client = session.client("service-quotas")
    account_id = session.client("sts").get_caller_identity()["Account"]
    print(f"Account {account_id}, region {session.region_name}\n")

    codes = quota_codes(client)
    low = 0
    for ladder, target in LADDER_TARGETS.items():
        print(ladder)
        for instance in INSTANCE_LADDERS[ladder]:
            low += not check_rung(client, codes, instance, target, apply)
        print()

    print(f"{low} rung(s) below target." if low else "All rungs meet their target.")
    return 1 if low else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually submit the increase requests")
    args = parser.parse_args()
    sys.exit(main(args.apply))
