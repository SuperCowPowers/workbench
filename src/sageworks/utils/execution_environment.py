"""ExecutionEnvironment provides logic/functionality to figure out the current execution environment"""

import os
import sys
import logging
import requests
from typing import Union
import boto3
from datetime import datetime, timezone

# SageWorks imports
from sageworks.utils.glue_utils import get_resolved_options

# Set up the logger
log = logging.getLogger("sageworks")


def running_on_glue():
    """
    Check if the current execution environment is an AWS Glue job.

    Returns:
        bool: True if running in AWS Glue environment, False otherwise.
    """
    # Check if GLUE_VERSION or GLUE_PYTHON_VERSION is in the environment
    if "GLUE_VERSION" in os.environ or "GLUE_PYTHON_VERSION" in os.environ:
        log.info("Running in AWS Glue Environment...")
        return True
    else:
        return False


def running_on_lambda():
    """
    Check if the current execution environment is an AWS Lambda function.

    Returns:
        bool: True if running in AWS Lambda environment, False otherwise.
    """
    if "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
        log.info("Running in AWS Lambda Environment...")
        return True
    else:
        return False


def running_on_docker() -> bool:
    """Check if the current environment is running on a Docker container.

    Returns:
        bool: True if running in a Docker container, False otherwise.
    """
    try:
        # Docker creates a .dockerenv file at the root of the directory tree inside the container.
        # If this file exists, it is very likely that we are running inside a Docker container.
        with open("/.dockerenv") as f:
            return True
    except FileNotFoundError:
        pass

    try:
        # Another method is to check the contents of /proc/self/cgroup which should be different
        # inside a Docker container.
        with open("/proc/self/cgroup") as f:
            if any("docker" in line for line in f):
                return True
    except FileNotFoundError:
        pass

    # Check if we are running on ECS
    if running_on_ecs():
        return True

    # Probably not running in a Docker container
    return False


def running_on_ecs() -> bool:
    """
    Check if the current environment is running on AWS ECS.

    Returns:
        bool: True if running on AWS ECS, False otherwise.
    """
    indicators = [
        "ECS_SERVICE_NAME",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_EXECUTION_ENV",
    ]
    return any(indicator in os.environ for indicator in indicators)


def running_as_service() -> bool:
    """
    Check if the current environment is running as a service (e.g. Docker, ECS, Glue, Lambda).

    Returns:
        bool: True if running as a service, False otherwise.
    """
    return running_on_docker() or running_on_glue() or running_on_lambda()


def _glue_job_from_script_name(args):
    """Get the Glue Job Name from the script name"""
    try:
        script_name = args["scriptLocation"]
        return os.path.splitext(os.path.basename(script_name))[0]
    except Exception:
        return "unknown"


def glue_job_name():
    """Get the Glue Job Name from the environment or script name"""
    # Define the required argument
    args = get_resolved_options(sys.argv)

    # Get the job name
    job_name = args.get("JOB_NAME") or _glue_job_from_script_name(args)
    return job_name


def glue_job_run_id(job_name: str, session: boto3.Session) -> Union[str, None]:
    """Retrieve the Glue Job Run ID closest to the current time for the given job name."""
    try:
        # Set current time in UTC
        current_time = datetime.now(timezone.utc)

        job_runs = session.client("glue").get_job_runs(JobName=job_name)
        if job_runs["JobRuns"]:
            # Find the job run with the StartedOn time closest to the current time
            closest_job_run = min(job_runs["JobRuns"], key=lambda run: abs(run["StartedOn"] - current_time))
            job_id = closest_job_run["Id"]
            return job_id[:9]  # Shorten the Job Run ID to 9 characters

        log.error(f"No runs found for Glue Job '{job_name}', returning None for Job Run ID.")
        return None

    except session.client("glue").exceptions.EntityNotFoundException:
        log.error(f"Glue Job '{job_name}' not found, returning None for Job Run ID.")
        return None
    except Exception as e:
        log.error(f"An error occurred while retrieving job run ID: {e}")
        return None


def ecs_job_name():
    """Get the ECS Job Name from the metadata endpoint or environment variables."""
    # Attempt to get the job name from ECS metadata
    ecs_metadata_uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")

    if ecs_metadata_uri:
        try:
            response = requests.get(f"{ecs_metadata_uri}/task")
            if response.status_code == 200:
                metadata = response.json()
                job_name = metadata.get("Family")  # 'Family' represents the ECS task definition family name
                if job_name:
                    return job_name
        except requests.RequestException as e:
            # Log the error or handle it as needed
            log.error(f"Failed to fetch ECS metadata: {e}")

    # Fallback to environment variables if metadata is not available
    job_name = os.environ.get("ECS_SERVICE_NAME", "unknown")
    return job_name


if __name__ == "__main__":
    """Test the Execution Environment utilities"""

    # Test running_on_glue
    assert running_on_glue() is False
    os.environ["GLUE_VERSION"] = "1.0"
    assert running_on_glue() is True
    del os.environ["GLUE_VERSION"]

    # Test running_on_lambda
    assert running_on_lambda() is False
    os.environ["AWS_LAMBDA_FUNCTION_NAME"] = "my_lambda_function"
    assert running_on_lambda() is True
    del os.environ["AWS_LAMBDA_FUNCTION_NAME"]

    # Test running_on_docker
    assert running_on_docker() is False
    os.environ["ECS_CONTAINER_METADATA_URI"] = "http://localhost:8080"
    assert running_on_docker() is True
    del os.environ["ECS_CONTAINER_METADATA_URI"]

    # Test running_on_ecs
    assert running_on_ecs() is False
    os.environ["ECS_CONTAINER_METADATA_URI"] = "http://localhost:8080"
    assert running_on_ecs() is True
    del os.environ["ECS_CONTAINER_METADATA_URI"]

    # Test getting the Glue Job Name
    print(glue_job_name())

    # Test getting the Glue Job Run ID
    from sageworks.core.cloud_platform.aws.aws_session import AWSSession

    session = AWSSession().boto3_session
    print(glue_job_run_id("Test_SageWorks_Shell", session))

    print("All tests passed!")
