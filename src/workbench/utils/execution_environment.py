"""ExecutionEnvironment provides logic/functionality to figure out the current execution environment"""

import os
import sys
import logging
import requests

# Workbench imports
from workbench.utils.glue_utils import get_resolved_options

# Set up the logger
log = logging.getLogger("workbench")


def running_on_glue() -> bool:
    """
    Check if the current execution environment is an AWS Glue job.

    Returns:
        bool: True if running in AWS Glue environment, False otherwise.
    """
    return bool(os.environ.get("GLUE_VERSION") or os.environ.get("GLUE_PYTHON_VERSION"))


def running_on_lambda() -> bool:
    """
    Check if the current execution environment is an AWS Lambda function.

    Returns:
        bool: True if running in AWS Lambda environment, False otherwise.
    """
    return bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


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
    return any(os.environ.get(indicator) for indicator in indicators)


def running_on_docker() -> bool:
    """Check if the current environment is running on a Docker container.

    Returns:
        bool: True if running in a Docker container, False otherwise.
    """
    # Check for .dockerenv file
    if os.path.exists("/.dockerenv"):
        return True

    # Check cgroup for docker
    try:
        with open("/proc/self/cgroup") as f:
            if any("docker" in line for line in f):
                return True
    except (FileNotFoundError, PermissionError):
        pass

    return running_on_ecs()


def running_as_service() -> bool:
    """
    Check if the current environment is running as a service (e.g. Docker, ECS, Glue, Lambda).

    Returns:
        bool: True if running as a service, False otherwise.
    """
    return any([running_on_glue(), running_on_lambda(), running_on_ecs(), running_on_docker()])


def glue_job_name() -> str:
    """Get Glue job name from environment or script."""
    args = get_resolved_options(sys.argv)

    if job_name := args.get("JOB_NAME"):
        return job_name

    # Fallback to script name
    if script_location := args.get("scriptLocation"):
        return os.path.splitext(os.path.basename(script_location))[0]

    return "unknown"


def ecs_job_name() -> str:
    """Get ECS job name from metadata or environment."""
    # Try metadata endpoint first
    if metadata_uri := os.environ.get("ECS_CONTAINER_METADATA_URI_V4"):
        try:
            response = requests.get(f"{metadata_uri}/task", timeout=5)
            if response.status_code == 200:
                if family := response.json().get("Family"):
                    return family
        except requests.RequestException as e:
            log.warning(f"Failed to fetch ECS metadata: {e}")

    return os.environ.get("ECS_SERVICE_NAME", "unknown")


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

    print("All tests passed!")
