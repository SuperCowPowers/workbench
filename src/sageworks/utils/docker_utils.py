"""Utilities for working with Docker containers."""

import os


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
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_EXECUTION_ENV",
    ]
    return any(indicator in os.environ for indicator in indicators)


if __name__ == "__main__":
    """Exercise the Docker Utils"""

    # Test the is_running_in_docker method
    print(f"Running in Docker: {running_on_docker()}")
    print(f"Running on ECS: {running_on_ecs()}")
