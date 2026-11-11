"""Estimate the spend AWS Partner Revenue Measurement (PRM) should attribute to Workbench in this account.

Splits the account's Cost Explorer usage spend (gross, before credits) into:
  tagged     driven by resources carrying the PRM tag (aws-apn-id-<partner-account>=pc:<product-code>)
             (SageMaker training jobs + endpoints, Batch/Fargate pipeline jobs)
  untagged   in PRM-included services but from resources without the tag
             (dashboard Fargate service, Redis, ALB, NAT, S3 storage, Athena, Glue, feature store, ...)
  excluded   not attributable by PRM (tax, support, VPC public IPs, S3 requests, third-party Bedrock models, ...)

The tagged figure is an estimate: SageMaker spend is split by usage-type family scaled by tag coverage,
and ECS spend is split by subtracting always-on services (task size x hours at Fargate list price) from
the total. When the PRM tag is an active cost allocation tag, the exact tag-filtered spend is printed too.

Scope is the caller's account and region (plus global and non-regional charges), matching the resource
inventory. Runs with the caller's own credentials (AWS_PROFILE); the Workbench role has no Cost Explorer access.

Usage:
    python scripts/admin/prm_revenue_estimate.py                         # last 3 complete months
    python scripts/admin/prm_revenue_estimate.py --start 2026-07 --months 3
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone

import boto3
from botocore.exceptions import ClientError

from workbench.utils.aws_utils import PRM_TAG_KEY, PRM_TAG_VALUE

# Gross usage line items; credits, refunds, tax, and fees are reported separately
USAGE_RECORD_TYPES = ["Usage", "DiscountedUsage", "SavingsPlanCoveredUsage"]

SAGEMAKER = "Amazon SageMaker"
ECS = "Amazon Elastic Container Service"
S3 = "Amazon Simple Storage Service"

# SageMaker usage-type families driven by tagged resources
SAGEMAKER_TRAINING = {"Train"}
SAGEMAKER_HOSTING = {"Host", "ServerlessInf", "AsyncInf"}

# Fargate Linux/x86 list price (us-east-1, us-west-2)
FARGATE_VCPU_HOUR = 0.04048
FARGATE_GB_HOUR = 0.004445

# Cost Explorer service names on the PRM Resource Tagging included-services list
PRM_INCLUDED = {
    SAGEMAKER,
    ECS,
    S3,
    "Amazon Elastic Compute Cloud - Compute",
    "EC2 - Other",
    "Amazon ElastiCache",
    "Amazon Elastic Load Balancing",
    "Amazon Athena",
    "AWS Glue",
    "AmazonCloudWatch",
    "AWS Lambda",
    "Amazon Simple Queue Service",
    "Amazon Simple Notification Service",
    "Amazon EC2 Container Registry (ECR)",
    "AWS Key Management Service",
    "AWS Secrets Manager",
    "AWS Step Functions",
    "Amazon DynamoDB",
    "Amazon Elastic File System",
    "Amazon Route 53",
    "AWS Backup",
    "Amazon API Gateway",
}

# Cost Explorer usage types carry a region prefix ('USW2-', 'EU-'), except many us-east-1 ones
REGION_PREFIX = re.compile(r"^[A-Z]{2,4}[0-9]?-")


def add_months(d: date, n: int) -> date:
    idx = d.year * 12 + d.month - 1 + n
    return date(idx // 12, idx % 12 + 1, 1)


def utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def usage_family(usage_type: str) -> str:
    """'USW2-Train:ml.g6.12xlarge' -> 'Train'"""
    return REGION_PREFIX.sub("", usage_type).split(":")[0]


def money(x: float) -> str:
    return f"${x:,.2f}"


def ce_rows(
    ce, start: date, end: date, group_by: list[str], scope: list[dict], filter_: dict | None = None
) -> list[dict]:
    """Monthly UnblendedCost rows {month, keys, cost} from Cost Explorer, limited to `scope` filters."""
    filters = scope + ([filter_] if filter_ else [])
    kwargs = {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": "MONTHLY",
        "Metrics": ["UnblendedCost"],
        "GroupBy": [{"Type": "DIMENSION", "Key": k} for k in group_by],
        "Filter": filters[0] if len(filters) == 1 else {"And": filters},
    }
    rows = []
    while True:
        resp = ce.get_cost_and_usage(**kwargs)
        for period in resp["ResultsByTime"]:
            month = period["TimePeriod"]["Start"][:7]
            for g in period["Groups"]:
                rows.append({"month": month, "keys": g["Keys"], "cost": float(g["Metrics"]["UnblendedCost"]["Amount"])})
        if not resp.get("NextPageToken"):
            return rows
        kwargs["NextPageToken"] = resp["NextPageToken"]


def tagged_arns(session) -> set[str]:
    """Lower-cased ARNs of every resource in the region carrying the PRM tag."""
    pages = session.client("resourcegroupstaggingapi").get_paginator("get_resources")
    tag_filter = [{"Key": PRM_TAG_KEY, "Values": [PRM_TAG_VALUE]}]
    return {
        r["ResourceARN"].lower()
        for page in pages.paginate(TagFilters=tag_filter)
        for r in page["ResourceTagMappingList"]
    }


def arn_type(arn: str) -> str:
    """'arn:aws:sagemaker:r:a:endpoint/x' -> 'sagemaker/endpoint'"""
    parts = arn.split(":")
    return f"{parts[2]}/{parts[5].split('/')[0]}"


def list_arns(sm, operation: str, list_key: str, arn_key: str, **kwargs) -> list[str]:
    pages = sm.get_paginator(operation).paginate(**kwargs)
    return [item[arn_key] for page in pages for item in page[list_key]]


def ecs_services(session) -> list[dict]:
    """Fargate ECS services (e.g. the dashboard) with their hourly cost and PRM tag status."""
    ecs = session.client("ecs")
    services = []
    for cluster in [c for page in ecs.get_paginator("list_clusters").paginate() for c in page["clusterArns"]]:
        arns = [a for page in ecs.get_paginator("list_services").paginate(cluster=cluster) for a in page["serviceArns"]]
        for i in range(0, len(arns), 10):
            for svc in ecs.describe_services(cluster=cluster, services=arns[i : i + 10], include=["TAGS"])["services"]:
                providers = {p["capacityProvider"] for p in svc.get("capacityProviderStrategy", [])}
                if svc.get("launchType") != "FARGATE" and not providers & {"FARGATE", "FARGATE_SPOT"}:
                    continue
                td = ecs.describe_task_definition(taskDefinition=svc["taskDefinition"])["taskDefinition"]
                task_hourly = int(td["cpu"]) / 1024 * FARGATE_VCPU_HOUR + int(td["memory"]) / 1024 * FARGATE_GB_HOUR
                tags = {t["key"]: t["value"] for t in svc.get("tags", [])}
                services.append(
                    {
                        "name": svc["serviceName"],
                        "arn": svc["serviceArn"],
                        "tagged": tags.get(PRM_TAG_KEY) == PRM_TAG_VALUE,
                        "hourly": task_hourly * svc["runningCount"],
                        "created": svc["createdAt"],
                    }
                )
    return services


def overlap_hours(created: datetime, lo: datetime, hi: datetime) -> float:
    hi = min(hi, datetime.now(timezone.utc))
    return max((hi - max(created, lo)).total_seconds() / 3600, 0.0)


def split_row(service: str, usage_type: str, cost: float, coverage: dict[str, float]) -> dict[str, float]:
    """Split one Cost Explorer line into tagged / untagged / excluded."""
    family = usage_family(usage_type)
    if service not in PRM_INCLUDED or (service == S3 and not family.startswith("TimedStorage")):
        return {"excluded": cost}
    if service == SAGEMAKER:
        frac = coverage["training"] if family in SAGEMAKER_TRAINING else 0.0
        frac = coverage["hosting"] if family in SAGEMAKER_HOSTING else frac
        return {"tagged": cost * frac, "untagged": cost * (1 - frac)}
    return {"untagged": cost}


def coverage_line(label: str, arns: list[str], tagged: set[str]) -> float:
    """Print 'tagged/total' for a resource list and return the tagged fraction."""
    untagged = [a.split("/")[-1] for a in arns if a.lower() not in tagged]
    n_tagged = len(arns) - len(untagged)
    extra = f"   untagged: {', '.join(untagged[:5])}{' ...' if len(untagged) > 5 else ''}" if untagged else ""
    print(f"  {label:<34}{n_tagged:>5}/{len(arns):<5}{extra}")
    return n_tagged / len(arns) if arns else 0.0


def tag_status(ce) -> str:
    """Cost allocation tag status of the PRM tag key."""
    try:
        tags = ce.list_cost_allocation_tags(TagKeys=[PRM_TAG_KEY])["CostAllocationTags"]
    except ClientError as e:
        return f"unknown ({e.response['Error']['Code']}; needs the management account)"
    return tags[0]["Status"] if tags else "not yet seen by Billing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="First month, YYYY-MM (default: --months complete months back)")
    parser.add_argument("--months", type=int, default=3, help="Number of months (default: 3)")
    args = parser.parse_args()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m").date()
    else:
        start = add_months(date.today().replace(day=1), -args.months)
    end = add_months(start, args.months)

    session = boto3.Session()
    account = session.client("sts").get_caller_identity()["Account"]
    ce = session.client("ce", region_name="us-east-1")
    sm = session.client("sagemaker")

    region = session.region_name
    scope = [
        {"Dimensions": {"Key": "LINKED_ACCOUNT", "Values": [account]}},
        {"Dimensions": {"Key": "REGION", "Values": [region, "global", "NoRegion"]}},
    ]
    print(f"Account {account} | region {region} (+ global) | {start:%Y-%m} to {add_months(end, -1):%Y-%m}")
    print(f"PRM tag {PRM_TAG_KEY}={PRM_TAG_VALUE}\n")

    # Tag coverage
    tagged = tagged_arns(session)
    services = ecs_services(session)
    print("Tag coverage")
    coverage = {
        "hosting": coverage_line(
            "SageMaker endpoints (current)", list_arns(sm, "list_endpoints", "Endpoints", "EndpointArn"), tagged
        ),
        "training": coverage_line(
            "SageMaker training jobs (window)",
            list_arns(
                sm,
                "list_training_jobs",
                "TrainingJobSummaries",
                "TrainingJobArn",
                CreationTimeAfter=utc(start),
                CreationTimeBefore=utc(end),
            ),
            tagged,
        ),
    }
    coverage_line(
        "SageMaker feature groups",
        list_arns(sm, "list_feature_groups", "FeatureGroupSummaries", "FeatureGroupArn"),
        tagged,
    )
    coverage_line("ECS services", [s["arn"] for s in services], tagged)
    by_type = Counter(arn_type(a) for a in tagged)
    print(f"  All tagged resources: {', '.join(f'{t} {n}' for t, n in by_type.most_common()) or 'none'}\n")

    # Spend split
    usage_filter = {"Dimensions": {"Key": "RECORD_TYPE", "Values": USAGE_RECORD_TYPES}}
    rows = ce_rows(ce, start, end, ["SERVICE", "USAGE_TYPE"], scope, usage_filter)
    by_month = defaultdict(lambda: defaultdict(float))
    by_service = defaultdict(lambda: defaultdict(float))
    sagemaker_families = defaultdict(float)
    ecs_by_month = defaultdict(float)
    for r in rows:
        service, usage_type = r["keys"]
        if service == ECS:
            ecs_by_month[r["month"]] += r["cost"]
            continue
        if service == SAGEMAKER:
            sagemaker_families[usage_family(usage_type)] += r["cost"]
        for bucket, cost in split_row(service, usage_type, r["cost"], coverage).items():
            by_month[r["month"]][bucket] += cost
            by_service[service][bucket] += cost

    # ECS: untagged always-on services estimated from task size x hours; Batch jobs are the remainder
    for month, total in ecs_by_month.items():
        lo = utc(datetime.strptime(month, "%Y-%m").date())
        hi = utc(add_months(lo.date(), 1))
        services_cost = sum(s["hourly"] * overlap_hours(s["created"], lo, hi) for s in services if not s["tagged"])
        untagged = min(services_cost, total)
        for bucket, cost in (("tagged", total - untagged), ("untagged", untagged)):
            by_month[month][bucket] += cost
            by_service[ECS][bucket] += cost

    buckets = ("tagged", "untagged", "excluded")
    header = f"  {'':<42}{'Total':>12}{'Tagged est':>12}{'Untagged':>12}{'Excluded':>12}"
    print("Usage spend by month (gross, before credits); Untagged = PRM-included service, resource lacks the tag")
    print(header)
    for month in sorted(by_month):
        vals = [by_month[month][b] for b in buckets]
        print(f"  {month:<42}{money(sum(vals)):>12}" + "".join(f"{money(v):>12}" for v in vals))
    totals = [sum(by_month[m][b] for m in by_month) for b in buckets]
    print(f"  {'TOTAL':<42}{money(sum(totals)):>12}" + "".join(f"{money(v):>12}" for v in totals) + "\n")

    print("By service (window)")
    print(header)
    for service, vals in sorted(by_service.items(), key=lambda kv: -sum(kv[1].values())):
        if sum(vals.values()) < 0.01:
            continue
        print(
            f"  {service[:42]:<42}{money(sum(vals.values())):>12}" + "".join(f"{money(vals[b]):>12}" for b in buckets)
        )
        if service == SAGEMAKER:
            for family, cost in sorted(sagemaker_families.items(), key=lambda kv: -kv[1]):
                if cost >= 0.01:
                    print(f"      {family[:38]:<38}{money(cost):>12}")
    print()

    # Credits and other record types: PRM may attribute net of credits
    record_types = defaultdict(float)
    for r in ce_rows(ce, start, end, ["RECORD_TYPE"], scope):
        record_types[r["keys"][0]] += r["cost"]
    print("Record types (window): " + " | ".join(f"{k} {money(v)}" for k, v in sorted(record_types.items())))

    # Exact tag-filtered spend, when Billing has the tag active
    status = tag_status(ce)
    print(f"Cost allocation tag {PRM_TAG_KEY}: {status}")
    if status == "Active":
        tag_filter = {"Tags": {"Key": PRM_TAG_KEY, "Values": [PRM_TAG_VALUE], "MatchOptions": ["EQUALS"]}}
        exact = defaultdict(float)
        for r in ce_rows(ce, start, end, ["SERVICE"], scope, {"And": [usage_filter, tag_filter]}):
            exact[r["keys"][0]] += r["cost"]
        print(f"  Exact tagged spend: {money(sum(exact.values()))}")
        for service, cost in sorted(exact.items(), key=lambda kv: -kv[1]):
            print(f"    {service[:40]:<40}{money(cost):>12}")
    else:
        print(
            "  Activate it in the management account (Billing > Cost allocation tags, with backfill) for exact numbers."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
