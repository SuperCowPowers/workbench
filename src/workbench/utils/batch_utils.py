"""Launch ad-hoc Python work onto AWS Batch (via SQS -> Lambda -> Batch)."""

import os
import re
import time
import logging
import tempfile
from datetime import datetime, timezone

log = logging.getLogger("workbench")

JOB_QUEUE = "workbench-job-queue"
_JOB_STATUSES = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]

# A Batch job that trains logs one line per poll from features_to_model, which is the only
# durable link back to SageMaker: nothing on the Batch job records what it submitted.
_BATCH_LOG_GROUP = "/aws/batch/job"
_TRAINING_JOB_RE = re.compile(r"Training job ([A-Za-z0-9._-]+) status")


def launch_batch(
    code: str,
    name: str,
    size: str = "small",
    script_args: list = None,
    realtime: bool = False,
) -> dict:
    """Write a Python script to a temp file and submit it to AWS Batch.

    The script runs as a standalone process in the Workbench image with the
    account's AWS credentials, so it can train and score at scale. It does **not**
    share the REPL namespace -- it must be self-contained (its own imports,
    explicit artifact names), and its results come back as Workbench artifacts (a
    new Model, FeatureSet, inference run), not as a returned value.

    See the `batch` guide for when to use it versus running inline.

    Args:
        code (str): The Python source to run on Batch. Self-contained.
        name (str): Script name; becomes the S3 key and the job label. Give it a
            clear, descriptive stem (e.g. "pxr_hpo_sweep").
        size (str, optional): Batch size tier -- "small" (default), "medium", or
            "large".
        script_args (list[str], optional): Args forwarded to the script as the
            PIPELINE_ARGS environment variable.
        realtime (bool, optional): Run with serverless=False. Defaults to
            serverless.

    Returns:
        dict: {"name", "size", "s3_path"} identifying the submitted job. The full
            submission log (message id, monitoring locations) is printed by the
            submitter.
    """
    from workbench.scripts.ml_pipeline_sqs import submit_to_sqs
    from workbench.utils.config_manager import ConfigManager

    if not name.endswith(".py"):
        name += ".py"

    # Fresh temp dir so the S3 key / job label is exactly `name` without collisions
    script_path = os.path.join(tempfile.mkdtemp(prefix="bosco_batch_"), name)
    with open(script_path, "w") as f:
        f.write(code)

    submit_to_sqs(script_path, size=size, realtime=realtime, script_args=script_args)

    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    return {"name": name, "size": size, "s3_path": f"s3://{bucket}/batch-jobs/{name}"}


def batch_jobs(name: str = None):
    """Recent AWS Batch jobs on the Workbench queue, newest first.

    Correlates with launch_batch: a job launched as `name="foo"` appears here as
    `workbench_foo_<timestamp>`, so pass a substring to find it.

    Notes:
        - A just-launched job takes a few seconds to appear (SQS -> Lambda ->
          Batch), so an empty result right after a launch is normal.
        - AWS keeps terminated jobs for a limited window (at least ~24h, often
          several days), so this is a recent view, not full history.

    Args:
        name (str, optional): Case-insensitive substring filter on the job name.

    Returns:
        pandas.DataFrame: columns [name, status, created, runtime, reason], sorted
            newest first. Empty if nothing matches.
    """
    import pandas as pd
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    batch = AWSAccountClamp().boto3_session.client("batch")

    rows = []
    for status in _JOB_STATUSES:
        for job in batch.list_jobs(jobQueue=JOB_QUEUE, jobStatus=status).get("jobSummaryList", []):
            started, stopped = job.get("startedAt"), job.get("stoppedAt")
            if started and stopped:
                runtime = f"{(stopped - started) / 1000:.0f}s"
            elif started:
                runtime = f"{(time.time() * 1000 - started) / 1000:.0f}s (running)"
            else:
                runtime = ""
            created = job.get("createdAt")
            rows.append(
                {
                    "name": job["jobName"],
                    "status": job["status"],
                    "created": pd.to_datetime(created, unit="ms") if created else pd.NaT,
                    "runtime": runtime,
                    "reason": job.get("statusReason", ""),
                }
            )

    df = pd.DataFrame(rows, columns=["name", "status", "created", "runtime", "reason"])
    if name:
        df = df[df["name"].str.contains(name, case=False, na=False)]
    return df.sort_values("created", ascending=False).reset_index(drop=True)


def _elapsed(since: datetime) -> str:
    """Compact h/m age string."""
    secs = (datetime.now(timezone.utc) - since).total_seconds()
    hours, minutes = divmod(int(secs // 60), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def batch_job_training_jobs(job_name: str) -> list:
    """SageMaker training jobs a Batch job submitted, in submission order.

    Recovered from the job's CloudWatch log rather than from AWS metadata: a Batch job
    records nothing about what it submitted, so the training job names are read back out
    of the lines features_to_model logs while it polls.

    A Batch job may submit any number of training jobs -- a script that builds several
    models trains several times, and plenty of Batch work never trains at all.

    Args:
        job_name (str): Batch job name, as it appears in batch_jobs().

    Returns:
        list[str]: Training job names, oldest first. Empty when the job trained nothing,
            has not started, or its logs have aged out.
    """
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    session = AWSAccountClamp().boto3_session
    batch, logs = session.client("batch"), session.client("logs")

    summaries = batch.list_jobs(jobQueue=JOB_QUEUE, filters=[{"name": "JOB_NAME", "values": [job_name]}])
    summaries = summaries.get("jobSummaryList", [])
    if not summaries:
        log.warning(f"No Batch job named {job_name!r} on {JOB_QUEUE} (AWS ages out terminated jobs)")
        return []

    job = batch.describe_jobs(jobs=[summaries[0]["jobId"]])["jobs"][0]
    stream = job.get("container", {}).get("logStreamName")
    if not stream:
        return []  # queued but never started, so there are no logs yet

    # Server-side filter: a long training run logs thousands of lines we don't want.
    names, token = [], None
    while True:
        page = logs.filter_log_events(
            logGroupName=_BATCH_LOG_GROUP,
            logStreamNames=[stream],
            filterPattern='"Training job"',
            **({"nextToken": token} if token else {}),
        )
        for event in page.get("events", []):
            match = _TRAINING_JOB_RE.search(event["message"])
            if match and match.group(1) not in names:
                names.append(match.group(1))
        token = page.get("nextToken")
        if not token:
            return names


def training_job_status(training_job_name: str) -> dict:
    """What a SageMaker training job is actually doing right now.

    `waiting` is the interesting field: a job in the Pending secondary status is queued
    for AWS capacity on its instance type and is burning wall-clock without training.

    Args:
        training_job_name (str): Full training job name, e.g. from batch_job_training_jobs().

    Returns:
        dict: {name, status, secondary_status, instance, age, in_status, message, waiting},
            or None if no training job by that name exists.
    """
    from botocore.exceptions import ClientError
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    sagemaker = AWSAccountClamp().boto3_session.client("sagemaker")
    try:
        job = sagemaker.describe_training_job(TrainingJobName=training_job_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            log.warning(f"No training job named {training_job_name!r}")
            return None
        raise

    transitions = job.get("SecondaryStatusTransitions") or []
    current = transitions[-1] if transitions else {}
    return {
        "name": job["TrainingJobName"],
        "status": job["TrainingJobStatus"],
        "secondary_status": job["SecondaryStatus"],
        "instance": job["ResourceConfig"]["InstanceType"],
        "age": _elapsed(job["CreationTime"]),
        "in_status": _elapsed(current["StartTime"]) if current.get("StartTime") else "n/a",
        "message": current.get("StatusMessage", ""),
        "waiting": job["SecondaryStatus"] == "Pending",
    }
