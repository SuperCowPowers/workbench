import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3
from workbench.utils.cloudwatch_utils import stream_log_events, print_log_event

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
    """Ensure the job definition exists with network configuration."""
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
            "networkConfiguration": {"assignPublicIp": "ENABLED"},  # Required for ECR image pull
        },
        timeout={"attemptDurationSeconds": 10800},  # 3 hours
    )

    log.info(f"Job definition ready: {name} (revision {response['revision']})")
    return name


def run_batch_job(script_path: str, stream_logs: bool = True) -> int:
    """Upload script, submit job, and track to completion.

    Args:
        script_path: Path to the ML pipeline script
        stream_logs: Whether to stream CloudWatch logs while job is running

    Returns:
        Exit code from the batch job
    """
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
                {"name": "ML_PIPELINE_S3_PATH", "value": s3_path},
                {"name": "WORKBENCH_BUCKET", "value": workbench_bucket},
            ]
        },
    )

    job_id = response["jobId"]
    log.info(f"Submitted job: {job_name} ({job_id})")

    # Track job to completion
    last_status = None
    log_stream_name = f"workbench-ml-pipeline-runner/default/{job_id}"
    last_event_time = None

    while True:
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = job["status"]

        if status != last_status:
            log.info(f"Job status: {status}")
            last_status = status

        # Stream logs when job is running and streaming is enabled
        if stream_logs and status == "RUNNING":
            try:
                # Stream new log events since last check
                event_count = 0
                for event in stream_log_events(
                        log_group_name="/aws/batch/job",
                        log_stream_name=log_stream_name,
                        start_time=last_event_time,
                        follow=False
                ):
                    print_log_event(event, show_stream=False, local_time=True)
                    last_event_time = datetime.fromtimestamp(event["timestamp"] / 1000 + 1, tz=datetime.now().astimezone().tzinfo)
                    event_count += 1

                if event_count == 0:
                    # No new events, wait a bit before checking again
                    time.sleep(5)
            except Exception as e:
                # Log stream might not exist yet, that's ok
                log.debug(f"Could not stream logs yet: {e}")
                time.sleep(5)

        if status in ["SUCCEEDED", "FAILED"]:
            # Stream any final log events
            if stream_logs:
                try:
                    log.info("Fetching final job output...")
                    for event in stream_log_events(
                            log_group_name="/aws/batch/job",
                            log_stream_name=log_stream_name,
                            start_time=last_event_time,
                            follow=False
                    ):
                        print_log_event(event, show_stream=False, local_time=True)
                except Exception as e:
                    log.debug(f"Could not fetch final logs: {e}")

            exit_code = job.get("attempts", [{}])[-1].get("exitCode", 1)
            if status == "FAILED":
                log.error(f"Job failed: {job.get('statusReason', 'Unknown reason')}")
            else:
                log.info(f"Job completed successfully")
            return exit_code

        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline script on AWS Batch")
    parser.add_argument("script_file", help="Local path to ML pipeline script")
    parser.add_argument("--no-stream", action="store_true", help="Don't stream CloudWatch logs")
    args = parser.parse_args()

    try:
        exit_code = run_batch_job(args.script_file, stream_logs=not args.no_stream)
        exit(exit_code)
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()