"""List in-progress SageMaker training jobs and what each one is actually doing.

Jobs sitting in the ``Pending`` secondary status are waiting on AWS capacity for
their instance type — they burn wall-clock without training. Those are listed
first and flagged.

Usage:
    python scripts/admin/training_job_status.py
    python scripts/admin/training_job_status.py --waiting     # only capacity-blocked jobs
    AWS_PROFILE=<profile> python scripts/admin/training_job_status.py
"""

import argparse
from datetime import datetime, timezone

import boto3


def in_progress_jobs(sm) -> list[dict]:
    """Full describe() payload for every training job currently InProgress."""
    names, token = [], None
    while True:
        kwargs = {"StatusEquals": "InProgress", "MaxResults": 100}
        if token:
            kwargs["NextToken"] = token
        resp = sm.list_training_jobs(**kwargs)
        names += [s["TrainingJobName"] for s in resp.get("TrainingJobSummaries", [])]
        token = resp.get("NextToken")
        if not token:
            break
    return [sm.describe_training_job(TrainingJobName=n) for n in names]


def elapsed(since: datetime) -> str:
    """Compact h/m age string."""
    secs = (datetime.now(timezone.utc) - since).total_seconds()
    hours, minutes = divmod(int(secs // 60), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def job_row(job: dict) -> dict:
    """Flatten a describe_training_job payload into the fields we print."""
    transitions = job.get("SecondaryStatusTransitions") or []
    current = transitions[-1] if transitions else {}
    return {
        "name": job["TrainingJobName"],
        "status": job["SecondaryStatus"],
        "instance": job["ResourceConfig"]["InstanceType"],
        "age": elapsed(job["CreationTime"]),
        "in_status": elapsed(current["StartTime"]) if current.get("StartTime") else "n/a",
        "message": current.get("StatusMessage", ""),
        "waiting": job["SecondaryStatus"] == "Pending",
    }


def main(waiting_only: bool):
    # Base identity, not the Workbench role: Workbench-ExecutionRole has no
    # sagemaker:ListTrainingJobs permission.
    sm = boto3.Session().client("sagemaker")
    rows = [job_row(j) for j in in_progress_jobs(sm)]
    if waiting_only:
        rows = [r for r in rows if r["waiting"]]
    if not rows:
        print("No matching in-progress training jobs.")
        return

    # Capacity-blocked jobs first, then longest-running
    rows.sort(key=lambda r: (not r["waiting"], r["name"]))

    hdr = f"{'':1} {'training job':<52} {'status':<12} {'instance':<18} {'age':>7} {'in status':>10}  message"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "!" if r["waiting"] else " "
        print(
            f"{flag} {r['name'][:52]:<52} {r['status']:<12} {r['instance']:<18} "
            f"{r['age']:>7} {r['in_status']:>10}  {r['message']}"
        )

    blocked = [r["name"] for r in rows if r["waiting"]]
    if blocked:
        print(f"\n{len(blocked)} job(s) waiting on capacity. Stop one with:")
        print(f"    aws sagemaker stop-training-job --training-job-name {blocked[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waiting", action="store_true", help="only show jobs waiting on capacity")
    args = parser.parse_args()
    main(args.waiting)
