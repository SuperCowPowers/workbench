"""ExecutionEnvironment provides logic/functionality to figure out the current execution environment"""

import os
import logging

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

    print("All tests passed!")
