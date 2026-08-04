"""Launch ad-hoc Python work onto AWS Batch (via SQS -> Lambda -> Batch)."""

import os
import re
import time
import atexit
import logging
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

# Workbench Imports
from workbench.utils.repl_utils import cprint_above_prompt, set_rprompt

log = logging.getLogger("workbench")

JOB_QUEUE = "workbench-job-queue"

# Every view in here is a "what is happening now" view, so all of them are scoped to this
# window. It also keeps the queries cheap: both AWS list calls scan history page by page.
LOOKBACK = timedelta(hours=48)

# Watcher cadence, and how long a launched job gets to appear on the queue at all
# (SQS -> Lambda -> Batch is normally seconds).
WATCH_INTERVAL = 300
APPEAR_TIMEOUT = 900
_TERMINAL_STATUS = {"SUCCEEDED", "FAILED"}

# Batch names a job `workbench_<script stem>_<%Y%m%d_%H%M%S>` (see ml_pipeline_batch).
_JOB_PREFIX = "workbench_"

# Watchers are daemon threads, so the interpreter exits without them; the event wakes
# them out of a poll interval first rather than tearing down mid-sleep.
_shutdown = threading.Event()
atexit.register(_shutdown.set)

# Finished jobs waiting to be reported into an agent turn. Bounded: nothing guarantees
# anyone ever drains it.
_completed = deque(maxlen=20)

# Jobs launched this session, in launch order, for the REPL's right-prompt lights.
_watched = {}
_LIGHT_TOKENS = {"running": "Darkyellow", "completed": "Lightgreen", "failed": "Red"}

# A Batch job that trains logs one line per poll from features_to_model, which is the only
# durable link back to SageMaker: nothing on the Batch job records what it submitted.
_BATCH_LOG_GROUP = "/aws/batch/job"
_TRAINING_JOB_RE = re.compile(r"Training job ([A-Za-z0-9._-]+) status")


def _client(service: str):
    """Boto3 client on the Workbench-assumed session. Imported lazily: the clamp pulls in core."""
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    return AWSAccountClamp().boto3_session.client(service)


def _lookback() -> datetime:
    """Start of the window these views cover."""
    return datetime.now(timezone.utc) - LOOKBACK


def _lookback_ms() -> str:
    """Start of the window as epoch milliseconds (what the Batch and Logs APIs take)."""
    return str(int(_lookback().timestamp() * 1000))


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
        size (str, optional): Container size for the script runner, which only
            orchestrates -- `to_model()` submits a SageMaker job that gets its own
            right-sized instance. Leave at "small" unless the script itself holds
            something big in memory.
        script_args (list[str], optional): Args forwarded to the script as the
            PIPELINE_ARGS environment variable.
        realtime (bool, optional): Run with serverless=False. Defaults to
            serverless.

    Returns:
        dict: {"name", "size", "s3_path"} identifying the submitted job. ``name`` is
            the stem, which is what `batch_jobs` matches on -- the full job name
            (``workbench_<stem>_<timestamp>``) is stamped downstream at submit time.
            The submission log (message id, monitoring locations) is printed by the
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

    stem = name[: -len(".py")]
    watch_batch_job(stem)

    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    return {"name": stem, "size": size, "s3_path": f"s3://{bucket}/batch-jobs/{name}"}


def watch_batch_job(name: str, interval: int = WATCH_INTERVAL) -> threading.Thread:
    """Poll a launched job in the background until it finishes, then report it.

    Reporting happens twice, because the launcher may be looking at neither: a banner
    is printed to the terminal, and the outcome is queued for `drain_completed`.

    Args:
        name (str): Job stem, as returned by `launch_batch` (matched as a substring).
        interval (int, optional): Seconds between polls. Defaults to WATCH_INTERVAL.

    Returns:
        threading.Thread: The started daemon thread.
    """
    _watched[name] = "running"
    thread = threading.Thread(target=_watch, args=(name, interval), daemon=True, name=f"batch-watch-{name}")
    thread.start()
    return thread


def _watch(name: str, interval: int) -> None:
    """Poll until the job reaches a terminal status, report it, and stop."""
    # The queue holds every job from the last 48 hours, so identity matters twice over.
    # The stem must run to the timestamp -- a bare prefix would let "pxr_reg_sweep" claim
    # "pxr_reg_sweep_v2" -- and the job must postdate this launch, since a re-run stem
    # has finished predecessors sitting in the same window.
    pattern = rf"^{_JOB_PREFIX}{re.escape(name)}_\d{{8}}_\d{{6}}$"
    launched = datetime.now(timezone.utc).replace(tzinfo=None)
    job_name = None
    waited = 0

    while not _shutdown.wait(interval):
        try:
            df = batch_jobs(job_name or name)
        except Exception as e:
            log.warning(f"Batch watcher for '{name}' could not list jobs: {e}")
            continue

        if job_name is None:
            # Oldest post-launch match is ours; anything newer is a separate launch.
            df = df[df["name"].str.match(pattern) & (df["created"] >= launched)]
            if df.empty:
                waited += interval
                if waited >= APPEAR_TIMEOUT:
                    _watched[name] = "failed"
                    _report({"name": name, "status": "NOT_FOUND", "reason": "never reached the Batch queue"})
                    return
                continue
            job_name = df.iloc[-1]["name"]

        row = df[df["name"] == job_name]
        if row.empty:
            continue

        row = row.iloc[0].to_dict()
        if row["status"] in _TERMINAL_STATUS:
            _watched[name] = "completed" if row["status"] == "SUCCEEDED" else "failed"
            _report(row)
            return


def _report(row: dict) -> None:
    """Announce a finished job and queue it for the next agent turn."""
    _completed.append(row)
    runtime = f" after {row['runtime']}" if row.get("runtime") else ""
    reason = f" -- {row['reason']}" if row.get("reason") else ""
    color = "lightgreen" if row["status"] == "SUCCEEDED" else "red"
    cprint_above_prompt(color, f"\nBatch job {row['name']} {row['status']}{runtime}{reason}")


def batch_lights():
    """`Batch [***]` -- a dot per job launched this session, for the right prompt.

    Yellow while a job runs, green when it succeeds, red when it fails or never
    reaches the queue. Reads the watchers' state, so it costs nothing per render.

    Returns:
        list[(Token, str)]: Pygments tokens, empty when nothing has been launched.
    """
    from pygments.token import Token

    if not _watched:
        return []
    dots = [(getattr(Token, _LIGHT_TOKENS[state]), "●") for state in _watched.values()]
    return [(Token.Grey, "Batch "), (Token.Blue, "[")] + dots + [(Token.Blue, "]")]


def install_batch_lights() -> None:
    """Put this session's Batch job lights on the right side of the REPL prompt."""
    from IPython import get_ipython

    set_rprompt(get_ipython(), batch_lights)


def drain_completed() -> list:
    """Pop every job that has finished since the last call.

    Returns:
        list[dict]: Job rows (name, status, runtime, reason), oldest first.
    """
    rows = []
    while _completed:
        rows.append(_completed.popleft())
    return rows


def batch_jobs(name: str = None):
    """AWS Batch jobs on the Workbench queue from the last LOOKBACK hours, newest first.

    Correlates with launch_batch: a job launched as `name="foo"` appears here as
    `workbench_foo_<timestamp>`, so pass a substring to find it.

    Note: a just-launched job takes a few seconds to appear (SQS -> Lambda -> Batch), so an
    empty result right after a launch is normal.

    Args:
        name (str, optional): Case-insensitive substring filter on the job name.

    Returns:
        pandas.DataFrame: columns [name, status, created, runtime, reason], sorted
            newest first. Empty if nothing matches.
    """
    import pandas as pd

    # An AFTER_CREATED_AT filter returns every status in one sweep (the API ignores
    # jobStatus whenever a filter is set), so the window replaces a per-status query.
    pages = (
        _client("batch")
        .get_paginator("list_jobs")
        .paginate(jobQueue=JOB_QUEUE, filters=[{"name": "AFTER_CREATED_AT", "values": [_lookback_ms()]}])
    )

    rows = []
    for page in pages:
        for job in page.get("jobSummaryList", []):
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
    batch, logs = _client("batch"), _client("logs")

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
            startTime=int(_lookback_ms()),
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

    try:
        job = _client("sagemaker").describe_training_job(TrainingJobName=training_job_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            log.warning(f"No training job named {training_job_name!r}")
            return None
        raise

    return _training_job_row(job)


def _training_job_row(job: dict) -> dict:
    """Flatten a describe_training_job payload into the status fields we report."""
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


def running_training_jobs():
    """Every SageMaker training job currently in progress, and what each is doing.

    The `waiting` column is the interesting one: those jobs are queued for AWS capacity on
    their instance type, holding no instance and making no progress, so a nightly run that
    never finishes usually has one of these behind it.

    Returns:
        pandas.DataFrame: training_job_status() fields as columns, capacity-blocked
            first, then by name. Empty if nothing is in progress.
    """
    import pandas as pd

    sagemaker = _client("sagemaker")
    # The status filter is applied per page, so an unbounded call walks the account's whole
    # training history a page at a time. CreationTimeAfter bounds the scan.
    pages = sagemaker.get_paginator("list_training_jobs").paginate(
        StatusEquals="InProgress",
        CreationTimeAfter=_lookback(),
        PaginationConfig={"PageSize": 100},
    )
    names = [summary["TrainingJobName"] for page in pages for summary in page["TrainingJobSummaries"]]

    df = pd.DataFrame([_training_job_row(sagemaker.describe_training_job(TrainingJobName=n)) for n in names])
    if df.empty:
        return df
    return df.sort_values(["waiting", "name"], ascending=[False, True]).reset_index(drop=True)
