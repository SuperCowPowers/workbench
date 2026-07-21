"""Launch ad-hoc Python work onto AWS Batch (via SQS -> Lambda -> Batch)."""

import os
import tempfile


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

    This is a real, billable compute launch. See the `batch` guide for when to use
    it versus running inline, and confirm with the user before calling it.

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
