import argparse
import os
import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3

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

    log.info(f"Wrapping {local_file_path}...")
    wrapped_script = glue_job_wrapper(local_file_path)
    log.info(f"Uploading wrapped script to {s3_path}...")
    upload_content_to_s3(wrapped_script, s3_path)
    return s3_path


def create_or_update_glue_job(s3_script_location: str, job_name: str):
    """Create or update a Glue job, and ensure a nightly trigger exists."""
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

    try:
        trigger_response = glue_client.create_trigger(
            Name=f"{job_name}_nightly_trigger",
            Type="SCHEDULED",
            Schedule="cron(0 0 * * ? *)",
            Actions=[{"JobName": job_name}],
            StartOnCreation=True,
        )
    except glue_client.exceptions.AlreadyExistsException:
        log.info(f"Trigger already exists: {job_name}_nightly_trigger")
        trigger_response = {"Name": f"{job_name}_nightly_trigger"}

    return {"job_name": job_name, "trigger_name": trigger_response["Name"]}


def main():
    parser = argparse.ArgumentParser(description="Create or update AWS Glue job with midnight schedule")
    parser.add_argument("script_file", help="Local path to script for the Glue job")
    args = parser.parse_args()

    s3_location = upload_script_to_s3(args.script_file)
    job_name = "workbench_" + os.path.basename(args.script_file).split(".")[0]

    result = create_or_update_glue_job(s3_location, job_name)
    print(f"Glue job '{result['job_name']}' set up with trigger '{result['trigger_name']}'")


if __name__ == "__main__":
    main()
