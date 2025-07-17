import argparse
import os
import logging
import time
import json
from typing import Dict, Any

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3
from workbench.utils.workbench_event_bridge import WorkbenchEventBridge

# Set up logging
log = logging.getLogger("workbench")
# Grab our Workbench S3 Bucket
cm = ConfigManager()
workbench_bucket = cm.get_config("WORKBENCH_BUCKET")


def glue_job_wrapper(local_file_path: str) -> str:
    """Wrap a local script file into a Workbench AWS Glue job."""
    glue_header = """
# Workbench GlueLaunch Header
import sys
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options
glue_args = get_resolved_options(sys.argv)
cm = ConfigManager()
cm.set_config("WORKBENCH_BUCKET", glue_args["workbench-bucket"])
# End Workbench GlueLaunch Header
"""
    with open(local_file_path, "r") as file:
        script_content = file.read()
    return f"{glue_header}\n\n{script_content}"


def upload_script_to_s3(local_file_path: str) -> str:
    """Upload local script file to S3 and return its S3 path."""
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"Script file not found: {local_file_path}")
    s3_path = f"s3://{workbench_bucket}/glue-jobs/{os.path.basename(local_file_path)}"
    log.info(f"Wrapping {local_file_path}…")
    wrapped_script = glue_job_wrapper(local_file_path)
    log.info(f"Uploading wrapped script to {s3_path}…")
    upload_content_to_s3(wrapped_script, s3_path)
    return s3_path


def create_or_update_glue_job(s3_script_location: str, job_name: str, create_trigger: bool = False):
    """Create or update a Glue job, and optionally ensure a nightly trigger exists."""
    glue_client = AWSAccountClamp().boto3_session.client("glue")
    job_args = {
        "Role": "Workbench-GlueRole",
        "Command": {"Name": "glueetl", "ScriptLocation": s3_script_location, "PythonVersion": "3"},
        "DefaultArguments": {
            "--job-language": "python",
            "--TempDir": "s3://aws-glue-temporary/",
            "--additional-python-modules": "workbench",
            "--workbench-bucket": workbench_bucket,
        },
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
        "MaxRetries": 0,
        "Timeout": 2880,
        "GlueVersion": "5.0",
        "MaxCapacity": 2.0,
    }
    try:
        glue_client.get_job(JobName=job_name)
        log.info(f"Updating existing Glue job: {job_name}")
        glue_client.update_job(JobName=job_name, JobUpdate=job_args)
    except glue_client.exceptions.EntityNotFoundException:
        log.info(f"Creating new Glue job: {job_name}")
        glue_client.create_job(Name=job_name, **job_args)
    result = {"job_name": job_name}

    # Only create trigger if requested
    if create_trigger:
        try:
            trigger_response = glue_client.create_trigger(
                Name=f"{job_name}_nightly_trigger",
                Type="SCHEDULED",
                Schedule="cron(0 0 * * ? *)",
                Actions=[{"JobName": job_name}],
                StartOnCreation=True,
            )
            result["trigger_name"] = trigger_response["Name"]
            log.info(f"Created trigger: {trigger_response['Name']}")
        except glue_client.exceptions.AlreadyExistsException:
            log.info(f"Trigger already exists: {job_name}_nightly_trigger")
            result["trigger_name"] = f"{job_name}_nightly_trigger"
    else:
        log.info("Skipping trigger creation (not requested)")
    return result


def get_detailed_job_info(job_name: str, job_run_id: str) -> Dict[str, Any]:
    """Get detailed information about a failed job including logs and metrics."""
    glue_client = AWSAccountClamp().boto3_session.client("glue")

    # Get job run details
    job_run = glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
    job_run_details = job_run["JobRun"]

    # Get job definition
    job_def = glue_client.get_job(JobName=job_name)
    return {
        "job_definition": job_def["Job"],
        "job_run_details": job_run_details,
        "allocated_capacity": job_run_details.get("AllocatedCapacity"),
        "max_capacity": job_run_details.get("MaxCapacity"),
        "worker_type": job_run_details.get("WorkerType"),
        "number_of_workers": job_run_details.get("NumberOfWorkers"),
        "dpu_seconds": job_run_details.get("DPUSeconds"),
        "execution_class": job_run_details.get("ExecutionClass"),
        "log_group_name": job_run_details.get("LogGroupName"),
        "predecessor_runs": job_run_details.get("PredecessorRuns"),
        "notification_property": job_run_details.get("NotificationProperty"),
        "glue_version": job_run_details.get("GlueVersion"),
    }


def run_glue_job(job_name: str) -> Dict[str, Any]:
    """Run a Glue job and track its progress."""
    glue_client = AWSAccountClamp().boto3_session.client("glue")
    log.info(f"Starting Glue job: {job_name}")
    # Start the job
    response = glue_client.start_job_run(JobName=job_name)
    job_run_id = response["JobRunId"]
    log.info(f"Job run started with ID: {job_run_id}")
    # Track job progress
    while True:
        job_run = glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
        job_state = job_run["JobRun"]["JobRunState"]
        log.info(f"Job {job_name} ({job_run_id}) state: {job_state}")
        if job_state in ["SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"]:
            break
        time.sleep(30)  # Check every 30 seconds
    # Get final job details
    final_job_run = glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
    job_run_details = final_job_run["JobRun"]
    # If job failed, get detailed information
    detailed_info = None
    if job_run_details["JobRunState"] != "SUCCEEDED":
        detailed_info = get_detailed_job_info(job_name, job_run_id)
    return {
        "job_name": job_name,
        "job_run_id": job_run_id,
        "state": job_run_details["JobRunState"],
        "exit_code": 0 if job_run_details["JobRunState"] == "SUCCEEDED" else 1,
        "start_time": job_run_details.get("StartedOn"),
        "end_time": job_run_details.get("CompletedOn"),
        "execution_time": job_run_details.get("ExecutionTime"),
        "error_message": job_run_details.get("ErrorMessage"),
        "detailed_failure_info": detailed_info,
    }


def emit_eventbridge_event(job_result: Dict[str, Any]):
    """Emit an EventBridge event with job execution results to workbench event bus."""
    event_detail = {
        "jobName": job_result["job_name"],
        "jobRunId": job_result["job_run_id"],
        "state": job_result["state"],
        "exitCode": job_result["exit_code"],
        "startTime": job_result["start_time"].isoformat() if job_result["start_time"] else None,
        "endTime": job_result["end_time"].isoformat() if job_result["end_time"] else None,
        "executionTime": job_result["execution_time"],
        "errorMessage": job_result["error_message"],
    }
    # Include detailed failure info if available
    if job_result.get("detailed_failure_info"):
        event_detail["detailedFailureInfo"] = job_result["detailed_failure_info"]
    try:
        event_bridge = WorkbenchEventBridge()
        response = event_bridge.send_event(event_detail, "Glue Job Execution Completed")
        if response:
            log.info(f"EventBridge event emitted successfully to workbench bus: {response}")
        else:
            log.warning("EventBridge event was not sent (bus may be unavailable)")
    except Exception as e:
        log.error(f"Failed to emit EventBridge event: {e}")


def cleanup_job_if_ephemeral(job_name: str, ephemeral: bool = False, job_succeeded: bool = True):
    """Delete the job if it was created as ephemeral and succeeded."""
    if not ephemeral:
        return
    if not job_succeeded:
        log.info(f"Job {job_name} failed - keeping for debugging (ephemeral flag ignored)")
        return
    glue_client = AWSAccountClamp().boto3_session.client("glue")
    try:
        log.info(f"Cleaning up ephemeral job: {job_name}")
        glue_client.delete_job(JobName=job_name)
        log.info(f"Ephemeral job {job_name} deleted successfully")
    except Exception as e:
        log.error(f"Failed to delete ephemeral job {job_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create or update AWS Glue job")
    parser.add_argument("script_file", help="Local path to script for the Glue job")
    parser.add_argument(
        "--run", action="store_true", default=False, help="Run the Glue job immediately after creation/update"
    )
    parser.add_argument(
        "--ephemeral", action="store_true", default=False, help="Delete the job after running (only works with --run)"
    )
    parser.add_argument("--trigger", action="store_true", default=False, help="Create a nightly trigger for the job")
    args = parser.parse_args()
    s3_location = upload_script_to_s3(args.script_file)
    job_name = "workbench_" + os.path.basename(args.script_file).split(".")[0]
    # Create or update the job
    result = create_or_update_glue_job(s3_location, job_name, args.trigger)
    if args.trigger:
        print(f"Glue job '{result['job_name']}' set up with trigger '{result['trigger_name']}'")
    else:
        print(f"Glue job '{result['job_name']}' created/updated (no trigger)")
    # Run the job if requested
    if args.run:
        try:
            job_result = run_glue_job(job_name)
            print(f"Job execution completed with state: {job_result['state']}")
            print(f"Exit code: {job_result['exit_code']}")
            # If job failed, dump detailed information
            if job_result["state"] != "SUCCEEDED":
                print("\n=== JOB FAILURE DETAILS ===")
                print(f"Error Message: {job_result['error_message']}")
                if job_result.get("detailed_failure_info"):
                    print(f"Failure info: {json.dumps(job_result['detailed_failure_info'], indent=2, default=str)}")
            # Emit EventBridge event
            emit_eventbridge_event(job_result)
            # Cleanup if ephemeral and successful
            job_succeeded = job_result["state"] == "SUCCEEDED"
            cleanup_job_if_ephemeral(job_name, args.ephemeral, job_succeeded)
            # Exit with the job's exit code
            exit(job_result["exit_code"])
        except Exception as e:
            log.error(f"Error running job: {e}")
            exit(1)


if __name__ == "__main__":
    main()
