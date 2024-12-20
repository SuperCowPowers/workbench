"""Resource utilities for Workbench"""

import sys
import importlib.resources as resources
import pathlib
import pkg_resources


def get_resource_path(package: str, resource: str) -> pathlib.Path:
    """Get the path to a resource file, compatible with Python 3.9 and higher.

    Args:
        package (str): The package where the resource is located.
        resource (str): The name of the resource file.

    Returns:
        pathlib.Path: The path to the resource file.
    """
    if sys.version_info >= (3, 10):
        # Python 3.10 and higher: use importlib.resources.path
        with resources.path(package, resource) as path:
            return path
    else:
        # Python 3.9 and lower: manually construct the path based on package location
        # Get the location of the installed package
        package_location = pathlib.Path(pkg_resources.get_distribution(package.split(".")[0]).location)
        resource_path = package_location / package.replace(".", "/") / resource

        if resource_path.exists():
            return resource_path
        else:
            raise FileNotFoundError(f"Resource '{resource}' not found in package '{package}'.")


if __name__ == "__main__":
    # Test the resource utilities
    with get_resource_path("workbench.resources", "open_source_api.key") as open_source_key_path:
        with open(open_source_key_path, "r") as key_file:
            print(key_file.read().strip())
