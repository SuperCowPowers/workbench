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


def ensure_job_definition():
    """Register or update the Batch job definition for ML pipeline runner."""
    batch = AWSAccountClamp().boto3_session.client("batch")
    name = "workbench-ml-pipeline-runner"
    response = batch.register_job_definition(
        jobDefinitionName=name,
        type="container",
        platformCapabilities=["FARGATE"],
        containerProperties={
            "image": get_ecr_image_uri(),
            "resourceRequirements": [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "4096"}],
            "jobRoleArn": get_batch_role_arn(),
            "executionRoleArn": get_batch_role_arn(),
            "environment": [
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
                {"name": "PYTHONUNBUFFERED", "value": "1"},
            ],
            # "networkConfiguration": {"assignPublicIp": "ENABLED"},  # Required for ECR Image Pull (when not in VPC)
        },
        timeout={"attemptDurationSeconds": 10800},  # 3 hours
    )
    log.info(f"Job definition ready: {name} (revision {response['revision']})")
    return name


def run_batch_job(script_path: str) -> int:
    """
    Submit and monitor an AWS Batch job for ML pipeline execution.
    This function:
    1. Uploads the ML pipeline script to S3
    2. Submits a Batch job to run the script in a container
    3. Monitors job status until completion
    4. Returns the job's exit code

    Args:
        script_path: Local path to the ML pipeline script

    Returns:
        Exit code from the batch job (0 for success, non-zero for failure)
    """
    batch = AWSAccountClamp().boto3_session.client("batch")
    script_name = Path(script_path).stem

    # Upload script to S3 for the container to download
    s3_path = f"s3://{workbench_bucket}/batch-jobs/{Path(script_path).name}"
    log.info(f"Uploading script to {s3_path}")
    upload_content_to_s3(Path(script_path).read_text(), s3_path)

    # Submit the Batch job
    job_name = f"workbench_{script_name}_{datetime.now():%Y%m%d_%H%M%S}"
    response = batch.submit_job(
        jobName=job_name,
        jobQueue="workbench-job-queue",
        jobDefinition=ensure_job_definition(),
        containerOverrides={
            "environment": [
                {"name": "ML_PIPELINE_S3_PATH", "value": s3_path},
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
            ]
        },
    )
    job_id = response["jobId"]
    log.info(f"Submitted job: {job_name} ({job_id})")

    # Monitor job execution
    last_status = None
    while True:
        # Check job status
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = job["status"]
        if status != last_status:
            log.info(f"Job status: {status}")
            last_status = status

        # Check if job completed
        if status in ["SUCCEEDED", "FAILED"]:
            exit_code = job.get("attempts", [{}])[-1].get("exitCode", 1)
            if status == "FAILED":
                log.error(f"Job failed: {job.get('statusReason', 'Unknown reason')}")
            else:
                log.info("Job completed successfully")

            # Get CloudWatch logs URL
            log_stream_name = job.get("container", {}).get("logStreamName")
            logs_url = get_cloudwatch_logs_url(log_group="/aws/batch/job", log_stream=log_stream_name)
            if logs_url:
                # OSC 8 hyperlink format for modern terminals
                clickable_url = f"\033]8;;{logs_url}\033\\{logs_url}\033]8;;\033\\"
                log.info(f"View logs: {clickable_url}")
            return exit_code

        # Sleep a bit before next status check
        time.sleep(10)


def main():
    """CLI entry point for running ML pipelines on AWS Batch."""
    parser = argparse.ArgumentParser(description="Run ML pipeline script on AWS Batch")
    parser.add_argument("script_file", help="Local path to ML pipeline script")
    args = parser.parse_args()
    try:
        exit_code = run_batch_job(args.script_file)
        exit(exit_code)
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
