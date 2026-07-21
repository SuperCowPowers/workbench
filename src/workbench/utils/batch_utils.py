"""Launch ad-hoc Python work onto AWS Batch (via SQS -> Lambda -> Batch)."""

import os
import time
import tempfile

JOB_QUEUE = "workbench-job-queue"
_JOB_STATUSES = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]


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
