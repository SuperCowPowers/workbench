import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3
from workbench.utils.cloudwatch_utils import get_cloudwatch_logs_url

log = logging.getLogger("workbench")
cm = ConfigManager()
workbench_bucket = cm.get_config("WORKBENCH_BUCKET")


def get_ecr_image_uri() -> str:
    """Get the ECR image URI for the current region."""
    region = AWSAccountClamp().region
    return f"507740646243.dkr.ecr.{region}.amazonaws.com/aws-ml-images/py312-ml-pipelines:0.1"


def get_batch_role_arn() -> str:
    """Get the Batch execution role ARN."""
    account_id = AWSAccountClamp().account_id
    return f"arn:aws:iam::{account_id}:role/Workbench-BatchRole"


def _log_cloudwatch_link(job: dict, message_prefix: str = "View logs") -> None:
    """
    Helper method to log CloudWatch logs link with clickable URL and full URL display.

    Args:
        job: Batch job description dictionary
        message_prefix: Prefix for the log message (default: "View logs")
    """
    log_stream = job.get("container", {}).get("logStreamName")
    logs_url = get_cloudwatch_logs_url(log_group="/aws/batch/job", log_stream=log_stream)
    if logs_url:
        clickable_url = f"\033]8;;{logs_url}\033\\{logs_url}\033]8;;\033\\"
        log.info(f"{message_prefix}: {clickable_url}")
    else:
        log.info("Check AWS Batch console for logs")


def run_batch_job(
    script_path: str,
    size: str = "small",
    realtime: bool = False,
    dt: bool = False,
    promote: bool = False,
    test_promote: bool = False,
) -> int:
    """
    Submit and monitor an AWS Batch job for ML pipeline execution.

    Uploads script to S3, submits Batch job, monitors until completion or 2 minutes of RUNNING.

    Args:
        script_path: Local path to the ML pipeline script
        size: Job size tier - "small" (default), "medium", or "large"
          - small: 2 vCPU, 4GB RAM for lightweight processing
          - medium: 4 vCPU, 8GB RAM for standard ML workloads
          - large: 8 vCPU, 16GB RAM for heavy training/inference
        realtime: If True, sets serverless=False for real-time processing (default: False)
        dt: If True, sets DT=True in environment (default: False)
        promote: If True, sets PROMOTE=True in environment (default: False)
        test_promote: If True, sets TEST_PROMOTE=True in environment (default: False)

    Returns:
        Exit code (0 for success/disconnected, non-zero for failure)
    """
    if size not in ["small", "medium", "large"]:
        raise ValueError(f"Invalid size '{size}'. Must be 'small', 'medium', or 'large'")

    batch = AWSAccountClamp().boto3_session.client("batch")
    script_name = Path(script_path).stem

    # Upload script to S3
    s3_path = f"s3://{workbench_bucket}/batch-jobs/{Path(script_path).name}"
    log.info(f"Uploading script to {s3_path}")
    upload_content_to_s3(Path(script_path).read_text(), s3_path)

    # Submit job
    job_name = f"workbench_{script_name}_{datetime.now():%Y%m%d_%H%M%S}"
    response = batch.submit_job(
        jobName=job_name,
        jobQueue="workbench-job-queue",
        jobDefinition=f"workbench-batch-{size}",
        containerOverrides={
            "environment": [
                {"name": "ML_PIPELINE_S3_PATH", "value": s3_path},
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
                {"name": "SERVERLESS", "value": "False" if realtime else "True"},
                {"name": "DT", "value": str(dt)},
                {"name": "PROMOTE", "value": str(promote)},
                {"name": "TEST_PROMOTE", "value": str(test_promote)},
            ]
        },
    )
    job_id = response["jobId"]
    log.info(f"Submitted job: {job_name} ({job_id}) using {size} tier")

    # Monitor job
    last_status, running_start = None, None
    while True:
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = job["status"]

        if status != last_status:
            log.info(f"Job status: {status}")
            last_status = status
            if status == "RUNNING":
                running_start = time.time()

        # Disconnect after 2 minutes of running
        if status == "RUNNING" and running_start and (time.time() - running_start >= 120):
            log.info("✅  ML Pipeline is running successfully!")
            _log_cloudwatch_link(job, "📊  Monitor logs")
            return 0

        # Handle completion
        if status in ["SUCCEEDED", "FAILED"]:
            exit_code = job.get("attempts", [{}])[-1].get("exitCode", 1)
            msg = (
                "Job completed successfully"
                if status == "SUCCEEDED"
                else f"Job failed: {job.get('statusReason', 'Unknown')}"
            )
            log.info(msg) if status == "SUCCEEDED" else log.error(msg)
            _log_cloudwatch_link(job)
            return exit_code

        time.sleep(10)


def main():
    """CLI entry point for running ML pipelines on AWS Batch."""
    parser = argparse.ArgumentParser(description="Run ML pipeline script on AWS Batch")
    parser.add_argument("script_file", help="Local path to ML pipeline script")
    parser.add_argument(
        "--size", default="small", choices=["small", "medium", "large"], help="Job size tier (default: small)"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Create realtime endpoints (default is serverless)",
    )
    parser.add_argument(
        "--dt",
        action="store_true",
        help="Set DT=True (models and endpoints will have '-dt' suffix)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Set Promote=True (models and endpoints will use promoted naming)",
    )
    parser.add_argument(
        "--test-promote",
        action="store_true",
        help="Set TEST_PROMOTE=True (creates test endpoint with '-test' suffix)",
    )
    args = parser.parse_args()
    try:
        exit_code = run_batch_job(
            args.script_file,
            size=args.size,
            realtime=args.realtime,
            dt=args.dt,
            promote=args.promote,
            test_promote=args.test_promote,
        )
        exit(exit_code)
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
