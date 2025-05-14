import boto3
import argparse
import os
from datetime import datetime
import uuid
import awswrangler as wr
import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3

# Set up logging
log = logging.getLogger("workbench")

def glue_job_wrapper(local_file_path: str) -> str:
    """
    Wrap a local script file into a Workbench AWS Glue job.

    Args:
        local_file_path (str): Local path to the script file.

    Returns:
       str: Wrapped script file contents
    """
    glue_header = f"""
# Glue Launch Header
import sys
from workbench.utils.config_manager import ConfigManager
from workbench.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the WORKBENCH_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("WORKBENCH_BUCKET", glue_args["workbench-bucket"])

# End: Glue Launch Header
"""
    # Read the local script file
    with open(local_file_path, 'r') as file:
        script_content = file.read()
    # Combine the header and the script content
    wrapped_script = f"{glue_header}\n{script_content}"
    return wrapped_script


def upload_script_to_s3(local_file_path) -> str:
    """Upload local file to the Workbench S3 bucket glue-jobs

    Args:
        local_file_path (str): Local path to the script file.

    Returns:
        str: S3 location of the uploaded file.
    """
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"Script file not found: {local_file_path}")

    # Get the S3 bucket name from the ConfigManager
    cm = ConfigManager()
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
    s3_path = f"s3://{workbench_bucket}/glue-jobs/{os.path.basename(local_file_path)}"

    # Wrap and Upload the script to S3
    log.info(f"Wrapping {local_file_path}...")
    wrapped_script = glue_job_wrapper(local_file_path)
    log.info(f"Uploading wrapped script to {s3_path}...")
    upload_content_to_s3(wrapped_script, s3_path)
    return s3_path


def create_glue_job(s3_script_location: str, job_name: str):
    """Create a Glue job that runs at midnight with the specified script

    Args:
        s3_script_location (str): S3 location of the script file.
        job_name (str): Name of the Glue job.
    """
    glue_client = AWSAccountClamp().boto3_session.client("glue")

    try:
        response = glue_client.create_job(
            Name=job_name,
            Role='Workbench-GlueRole',
            Command={
                'Name': 'glueetl',
                'ScriptLocation': s3_script_location,
                'PythonVersion': '3'
            },
            DefaultArguments={
                '--job-language': 'python',
                '--TempDir': 's3://aws-glue-temporary/',
                '--additional-python-modules': 'workbench'
            },
            ExecutionProperty={
                'MaxConcurrentRuns': 1
            },
            MaxRetries=0,
            Timeout=2880,
            GlueVersion='5.0'
        )

        # Set up the trigger to run the job every night at midnight
        trigger_response = glue_client.create_trigger(
            Name=f"{job_name}-nightly-trigger",
            Type='SCHEDULED',
            Schedule="cron(0 0 * * ? *)",
            Actions=[{
                'JobName': job_name
            }],
            StartOnCreation=True
        )

        return {
            'job_name': job_name,
            'job_run_id': response['Name'],
            'trigger_name': trigger_response['Name']
        }

    except Exception as e:
        print(f"Error creating Glue job: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Create AWS Glue job with midnight schedule')
    parser.add_argument('script_file', help='Local path to script for the Glue job')
    args = parser.parse_args()

    # Upload local file to S3
    s3_location = upload_script_to_s3(args.script_file)

    # Generate a job name based on the script name (stripping the extension)
    job_name = os.path.basename(args.script_file).split('.')[0]

    # Create our Glue job
    result = create_glue_job(s3_location, job_name)

    print(f"Successfully created Glue job '{result['job_name']}' with midnight trigger")


if __name__ == "__main__":
    main()