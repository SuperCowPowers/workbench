"""Model Utilities for Workbench models"""

import logging
import pandas as pd
import importlib.resources
from pathlib import Path

# Set up the log
log = logging.getLogger("workbench")


def model_instance_info() -> pd.DataFrame:
    """Get the instance information for the Model"""
    data = [
        {
            "Instance Name": "ml.t2.medium",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.06,
            "Category": "General",
            "Architecture": "x86_64",
        },
        {
            "Instance Name": "ml.m7i.large",
            "vCPUs": 2,
            "Memory": 8,
            "Price per Hour": 0.12,
            "Category": "General",
            "Architecture": "x86_64",
        },
        {
            "Instance Name": "ml.c7i.large",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.11,
            "Category": "Compute",
            "Architecture": "x86_64",
        },
        {
            "Instance Name": "ml.c7i.xlarge",
            "vCPUs": 4,
            "Memory": 8,
            "Price per Hour": 0.21,
            "Category": "Compute",
            "Architecture": "x86_64",
        },
        {
            "Instance Name": "ml.c7g.large",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.09,
            "Category": "Compute",
            "Architecture": "arm64",
        },
        {
            "Instance Name": "ml.c7g.xlarge",
            "vCPUs": 4,
            "Memory": 8,
            "Price per Hour": 0.17,
            "Category": "Compute",
            "Architecture": "arm64",
        },
    ]
    return pd.DataFrame(data)


def instance_architecutre(instance_name: str) -> str:
    """Get the architecture for the given instance name"""
    info = model_instance_info()
    return info[info["Instance Name"] == instance_name]["Architecture"].values[0]


def supported_instance_types(arch: str = "x86_64") -> list:
    """Get the supported instance types for the Model/Model"""

    # Filter the instance types based on the architecture
    info = model_instance_info()
    return info[info["Architecture"] == arch]["Instance Name"].tolist()


def get_custom_script_path(package: str, script_name: str) -> Path:
    with importlib.resources.path(f"workbench.model_scripts.custom_models.{package}", script_name) as script_path:
        return script_path


if __name__ == "__main__":
    """Exercise the Model Utilities"""

    # Get the instance information
    print(model_instance_info())

    # Get the supported instance types
    print(supported_instance_types())

    # Get the architecture for the given instance
    print(instance_architecutre("ml.c7i.large"))
    print(instance_architecutre("ml.c7g.large"))

    # Get the custom script path
    print(get_custom_script_path("chem_info", "molecular_descriptors.py"))
