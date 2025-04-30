"""Model Utilities for Workbench models"""

import logging
import pandas as pd
import importlib.resources
from pathlib import Path
import os
import json
import tempfile
import tarfile
import awswrangler as wr
from typing import Optional

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


def instance_architecture(instance_name: str) -> str:
    """Get the architecture for the given instance name"""
    info = model_instance_info()
    return info[info["Instance Name"] == instance_name]["Architecture"].values[0]


def supported_instance_types(arch: str = "x86_64") -> list:
    """Get the supported instance types for the Model/Model"""

    # Filter the instance types based on the architecture
    info = model_instance_info()
    return info[info["Architecture"] == arch]["Instance Name"].tolist()


def get_custom_script_path(package: str, script_name: str) -> Path:
    package_path = importlib.resources.files(f"workbench.model_scripts.custom_models.{package}")
    script_path = package_path / script_name
    return script_path


def proximity_model(model: "Model", prox_model_name: str, track_columns: list = None) -> "Model":
    """Create a proximity model based on the given model

    Args:
        model (Model): The model to create the proximity model from
        prox_model_name (str): The name of the proximity model to create
        track_columns (list, optional): List of columns to track in the proximity model
    Returns:
        Model: The proximity model
    """
    from workbench.api import Model, ModelType, FeatureSet  # noqa: F401 (avoid circular import)

    # Get the custom script path for the proximity model
    script_path = get_custom_script_path("proximity", "feature_space_proximity.template")

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Create the Proximity Model from our FeatureSet
    fs = FeatureSet(model.get_input())
    prox_model = fs.to_model(
        name=prox_model_name,
        model_type=ModelType.TRANSFORMER,
        feature_list=features,
        target_column=target,
        description=f"Proximity Model for {model.uuid}",
        tags=["proximity", model.uuid],
        custom_script=script_path,
        custom_args={"track_columns": track_columns},
    )
    return prox_model


def load_category_mappings_from_s3(model_artifact_uri: str) -> Optional[dict]:
    """
    Download and extract category mappings from a model artifact in S3.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact.

    Returns:
        dict: The loaded category mappings or None if not found.
    """
    category_mappings = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model artifact
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

        # Extract tarball
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=tmpdir, filter="data")

        # Look for category mappings in base directory only
        mappings_path = os.path.join(tmpdir, "category_mappings.json")

        if os.path.exists(mappings_path):
            try:
                with open(mappings_path, "r") as f:
                    category_mappings = json.load(f)
                print(f"Loaded category mappings from {mappings_path}")
            except Exception as e:
                print(f"Failed to load category mappings from {mappings_path}: {e}")

    return category_mappings


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model

    # Get the instance information
    print(model_instance_info())

    # Get the supported instance types
    print(supported_instance_types())

    # Get the architecture for the given instance
    print(instance_architecture("ml.c7i.large"))
    print(instance_architecture("ml.c7g.large"))

    # Get the custom script path
    print(get_custom_script_path("chem_info", "molecular_descriptors.py"))

    # Test the proximity model
    m = Model("abalone-regression")
    prox_model = proximity_model(m, "abalone-prox")
    print(prox_model)
