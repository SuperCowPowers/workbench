import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3

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
    """Ensure the job definition exists (creates or updates)."""
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
            "environment": [{"name": "WORKBENCH_BUCKET", "value": workbench_bucket}],
        },
        timeout={"attemptDurationSeconds": 10800},  # 3 hours
    )
    log.info(f"Job definition ready: {name} (revision {response['revision']})")
    return name


def run_batch_job(script_path: str) -> int:
    """Upload script, submit job, and track to completion."""
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
        jobDefinition=ensure_job_definition(),
        containerOverrides={
            "environment": [
                {"name": "SCRIPT_S3_PATH", "value": s3_path},
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
            ]
        },
    )

    job_id = response["jobId"]
    log.info(f"Submitted job: {job_name} ({job_id})")

    # Track job to completion
    while True:
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = job["status"]
        log.info(f"Job status: {status}")

        if status in ["SUCCEEDED", "FAILED"]:
            exit_code = job.get("attempts", [{}])[-1].get("exitCode", 1)
            if status == "FAILED":
                log.error(f"Job failed: {job.get('statusReason', 'Unknown reason')}")
            return exit_code

        time.sleep(30)


def main():
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
