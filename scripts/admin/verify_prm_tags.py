"""Verify AWS Partner Revenue Measurement (PRM) tagging across a Workbench account.

Checks that the PRM tag is on every resource that drives Workbench spend: SageMaker endpoints and
feature groups, the latest Batch job definitions (with tag propagation), the dashboard ECS service and
its running tasks, the Workbench S3 bucket, and the ECS tasks of running Batch jobs. Also lists every
tagged resource by type. Exits 1 when anything is missing.

Runs with the caller's own credentials (AWS_PROFILE, an admin profile); the Workbench role has no
access to the tagging API or bucket tags.

Usage:
    python scripts/admin/verify_prm_tags.py
"""

import collections
import sys
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import ClientError

from workbench.utils.aws_utils import PRM_TAG_KEY, PRM_TAG_VALUE
from workbench.utils.config_manager import ConfigManager


def has_tag(tags: dict) -> bool:
    return tags.get(PRM_TAG_KEY) == PRM_TAG_VALUE


def ecs_tags(resource: dict) -> dict:
    return {t["key"]: t["value"] for t in resource.get("tags", [])}


def main() -> int:
    session = boto3.Session()
    account = session.client("sts").get_caller_identity()["Account"]
    print(f"Account {account} | {session.region_name} | {PRM_TAG_KEY}={PRM_TAG_VALUE}\n")
    gaps = []

    def check(ok: bool, label: str, hint: str = ""):
        print(f"  {'ok ' if ok else 'MISSING'}  {label}")
        if not ok:
            gaps.append(f"{label}{f' ({hint})' if hint else ''}")

    # Every resource carrying the tag
    pages = session.client("resourcegroupstaggingapi").get_paginator("get_resources")
    tag_filter = [{"Key": PRM_TAG_KEY, "Values": [PRM_TAG_VALUE]}]
    tagged = {
        r["ResourceARN"].lower() for p in pages.paginate(TagFilters=tag_filter) for r in p["ResourceTagMappingList"]
    }
    types = collections.Counter(f"{a.split(':')[2]}/{a.split(':')[5].split('/')[0]}" for a in tagged)
    print("Tagged resources: " + ", ".join(f"{t} {n}" for t, n in sorted(types.items())) + "\n")

    # SageMaker endpoints and feature groups
    sm = session.client("sagemaker")
    endpoints = [e["EndpointArn"] for p in sm.get_paginator("list_endpoints").paginate() for e in p["Endpoints"]]
    groups = sm.get_paginator("list_feature_groups").paginate()
    feature_groups = [g["FeatureGroupArn"] for p in groups for g in p["FeatureGroupSummaries"]]
    for label, arns in (("endpoints", endpoints), ("feature groups", feature_groups)):
        n = sum(a.lower() in tagged for a in arns)
        check(n == len(arns), f"SageMaker {label} {n}/{len(arns)}", "run backfill_prm_tags.py --apply")

    # Latest Batch job definitions: tagged, and propagating tags onto the ECS tasks
    batch = session.client("batch")
    latest = {}
    for page in batch.get_paginator("describe_job_definitions").paginate(status="ACTIVE"):
        for jd in page["jobDefinitions"]:
            name = jd["jobDefinitionName"]
            if name.startswith("workbench-batch-") and jd["revision"] > latest.get(name, {}).get("revision", -1):
                latest[name] = jd
    for name, jd in sorted(latest.items()):
        ok = jd.get("propagateTags") and has_tag(jd.get("tags", {}))
        check(ok, f"Batch job definition {name}:{jd['revision']}", "cdk deploy workbench_compute")

    # Dashboard ECS service and its running tasks (tasks pick up tags only at launch)
    ecs = session.client("ecs")
    for cluster in ecs.list_clusters()["clusterArns"]:
        for service_arn in ecs.list_services(cluster=cluster)["serviceArns"]:
            svc = ecs.describe_services(cluster=cluster, services=[service_arn], include=["TAGS"])["services"][0]
            ok = svc.get("propagateTags") == "SERVICE" and has_tag(ecs_tags(svc))
            check(ok, f"ECS service {svc['serviceName']}", "cdk deploy workbench_dashboard")
            task_arns = ecs.list_tasks(cluster=cluster, serviceName=svc["serviceName"])["taskArns"]
            tasks = ecs.describe_tasks(cluster=cluster, tasks=task_arns, include=["TAGS"])["tasks"] if task_arns else []
            for task in tasks:
                check(
                    has_tag(ecs_tags(task)),
                    f"  task {task['taskArn'].split('/')[-1][:12]} started {task.get('startedAt')}",
                    "aws ecs update-service --force-new-deployment",
                )

    # Workbench S3 bucket (not CDK-created)
    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    try:
        tag_set = session.client("s3").get_bucket_tagging(Bucket=bucket)["TagSet"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchTagSet":
            raise
        tag_set = []
    bucket_tags = {t["Key"]: t["Value"] for t in tag_set}
    check(has_tag(bucket_tags), f"S3 bucket {bucket}", "aws resourcegroupstaggingapi tag-resources")

    # ECS tasks of Batch jobs running now (jobs submitted before a job-definition update keep the old revision)
    since = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp() * 1000)
    created_filter = [{"name": "AFTER_CREATED_AT", "values": [str(since)]}]
    summaries = batch.list_jobs(jobQueue="workbench-job-queue", filters=created_filter)["jobSummaryList"]
    running = [j["jobId"] for j in summaries if j["status"] == "RUNNING"][:100]
    jobs = batch.describe_jobs(jobs=running)["jobs"] if running else []
    print(f"\n  {len(jobs)} Batch job(s) running")
    for job in jobs:
        task_arn = (job.get("container") or {}).get("taskArn")
        if not task_arn:
            continue
        cluster = task_arn.split(":task/")[1].split("/")[0]
        task = ecs.describe_tasks(cluster=cluster, tasks=[task_arn], include=["TAGS"])["tasks"]
        name, revision = job["jobDefinition"].split("/")[-1].rsplit(":", 1)
        current = latest.get(name, {}).get("revision") == int(revision)
        ok = bool(task) and has_tag(ecs_tags(task[0]))
        if current:
            check(ok, f"  Batch task {job['jobName']} ({name}:{revision})", "job definition not propagating tags")
        else:
            print(f"  info     Batch task {job['jobName']} on older {name}:{revision} (tagged={ok})")

    print(f"\n{'All PRM tags in place.' if not gaps else f'{len(gaps)} gap(s):'}")
    for gap in gaps:
        print(f"  - {gap}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
