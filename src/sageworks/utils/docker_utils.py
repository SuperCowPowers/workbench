"""Utilities for working with Docker containers."""


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
            return any("docker" in line for line in f)
    except FileNotFoundError:
        pass

    return False


if __name__ == "__main__":
    """Exercise the Docker Utils"""

    # Test the is_running_in_docker method
    print(f"Running in Docker: {running_on_docker()}")
