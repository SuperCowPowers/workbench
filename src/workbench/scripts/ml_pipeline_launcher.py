import argparse
import os
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3

# Set up logging
log = logging.getLogger("workbench")

# Grab our Workbench S3 Bucket and Account Information
cm = ConfigManager()
workbench_bucket = cm.get_config("WORKBENCH_BUCKET")


def get_ecr_image_uri() -> str:
    """Get the ECR image URI for the current region."""
    workbench_ecr_account_id = "507740646243"
    region = AWSAccountClamp().region
    return f"{workbench_ecr_account_id}.dkr.ecr.{region}.amazonaws.com/aws-ml-images/py312-ml-pipelines:0.1"


def get_job_role_arn() -> str:
    """Get the Batch execution role ARN."""
    account_id = AWSAccountClamp().account_id
    return f"arn:aws:iam::{account_id}:role/Workbench-BatchRole"


def upload_script_to_s3(local_file_path: str) -> str:
    """Upload local script file to S3 and return its S3 path."""
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"Script file not found: {local_file_path}")

    s3_path = f"s3://{workbench_bucket}/batch-jobs/{os.path.basename(local_file_path)}"
    log.info(f"Uploading script to {s3_path}...")

    with open(local_file_path, "r") as file:
        script_content = file.read()

    upload_content_to_s3(script_content, s3_path)
    return s3_path


def create_or_update_batch_job_definition():
    """Create or update the reusable Batch job definition."""
    batch_client = AWSAccountClamp().boto3_session.client("batch")
    job_definition_name = "workbench-ml-pipeline-runner"

    job_definition = {
        "jobDefinitionName": job_definition_name,
        "type": "container",
        "containerProperties": {
            "image": get_ecr_image_uri(),
            "vcpus": 2,
            "memory": 4096,
            "jobRoleArn": get_job_role_arn(),
            "environment": [{"name": "WORKBENCH_BUCKET", "value": workbench_bucket}],
        },
        "retryStrategy": {"attempts": 1},
        "timeout": {"attemptDurationSeconds": 10800},  # 3 hours
    }

    try:
        # Check if job definition exists
        response = batch_client.describe_job_definitions(jobDefinitionName=job_definition_name, status="ACTIVE")
        if response["jobDefinitions"]:
            log.info(f"Updating job definition: {job_definition_name}")
        else:
            log.info(f"No active job definition found, creating: {job_definition_name}")
    except batch_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "InvalidParameterValueException":
            log.info(f"Creating new job definition: {job_definition_name}")
        else:
            log.error(f"Error checking job definition: {e}")
            raise
    except Exception as e:
        log.error(f"Unexpected error checking job definition: {e}")
        raise

    # Register (create or update) the job definition
    batch_client.register_job_definition(**job_definition)
    return job_definition_name


def submit_batch_job(s3_script_path: str, script_name: str) -> Tuple[str, str]:
    """Submit a batch job and return the job ID."""
    batch_client = AWSAccountClamp().boto3_session.client("batch")

    # Create unique job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    job_name = f"workbench_{script_name}_{timestamp}_{unique_id}"

    # Ensure job definition exists
    job_definition_name = create_or_update_batch_job_definition()

    job_submission = {
        "jobName": job_name,
        "jobQueue": "workbench-job-queue",  # Placeholder - assume this exists
        "jobDefinition": job_definition_name,
        "containerOverrides": {
            "environment": [
                {"name": "SCRIPT_S3_PATH", "value": s3_script_path},
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
            ]
        },
    }

    log.info(f"Submitting batch job: {job_name}")
    response = batch_client.submit_job(**job_submission)
    job_id = response["jobId"]
    log.info(f"Job submitted with ID: {job_id}")
    return job_id, job_name


def track_batch_job(job_id: str, job_name: str) -> Dict[str, Any]:
    """Track batch job progress and return final status."""
    batch_client = AWSAccountClamp().boto3_session.client("batch")

    log.info(f"Tracking job: {job_name} ({job_id})")

    while True:
        response = batch_client.describe_jobs(jobs=[job_id])
        job_detail = response["jobs"][0]
        job_status = job_detail["jobStatus"]

        log.info(f"Job {job_name} ({job_id}) status: {job_status}")

        if job_status in ["SUCCEEDED", "FAILED"]:
            break

        time.sleep(30)  # Check every 30 seconds

    # Get final job details
    final_response = batch_client.describe_jobs(jobs=[job_id])
    final_job_detail = final_response["jobs"][0]

    return {
        "job_name": job_name,
        "job_id": job_id,
        "status": final_job_detail["jobStatus"],
        "exit_code": final_job_detail.get("attempts", [{}])[-1].get("exitCode", 1),
        "start_time": final_job_detail.get("startedAt"),
        "end_time": final_job_detail.get("stoppedAt"),
        "status_reason": final_job_detail.get("statusReason"),
        "job_detail": final_job_detail,
    }


def run_batch_job(local_file_path: str) -> Dict[str, Any]:
    """Main function to upload script and run batch job."""
    script_name = os.path.basename(local_file_path).split(".")[0]

    # Upload script to S3
    s3_script_path = upload_script_to_s3(local_file_path)

    # Submit job
    job_id, job_name = submit_batch_job(s3_script_path, script_name)

    # Track progress
    job_result = track_batch_job(job_id, job_name)

    return job_result


def main():
    parser = argparse.ArgumentParser(description="Create and run an AWS Batch job for ML pipeline")
    parser.add_argument("script_file", help="Local path to ML pipeline script")
    parser.add_argument("--run", action="store_true", default=True, help="Run the job immediately")
    args = parser.parse_args()

    try:
        job_result = run_batch_job(args.script_file)
        print(f"Job execution completed with status: {job_result['status']}")
        print(f"Exit code: {job_result['exit_code']}")

        if job_result["status"] != "SUCCEEDED":
            print(f"Status reason: {job_result['status_reason']}")

        exit(job_result["exit_code"])

    except Exception as e:
        log.error(f"Error running batch job: {e}")
        exit(1)


if __name__ == "__main__":
    main()
