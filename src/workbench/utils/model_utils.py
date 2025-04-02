"""Model Utilities for Workbench models"""

import logging
import pandas as pd
import importlib.resources
from pathlib import Path
import os
import tempfile
import tarfile
import awswrangler as wr
from typing import Optional, List, Tuple

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
    with importlib.resources.path(f"workbench.model_scripts.custom_models.{package}", script_name) as script_path:
        return script_path


def proximity_model(model: "Model", prox_model_name: str, shap_50: bool = True) -> "Model":
    """Create a proximity model based on the given model

    Args:
        model (Model): The model to create the proximity model from
        prox_model_name (str): The name of the proximity model to create
        shap_50 (bool): Whether to use the top 50 SHAP features for the proximity model
    Returns:
        Model: The proximity model
    """
    from workbench.api import Model, ModelType, FeatureSet  # noqa: F401 (avoid circular import)
    from workbench.utils.shap_utils import shap_feature_importance  # noqa: F401 (avoid circular import)

    # Get the custom script path for the proximity model
    script_path = get_custom_script_path("proximity", "feature_space_proximity.template")

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Compute the top 50 Shap feature importances?
    if shap_50:
        shap_importances = shap_feature_importance(model, top_n=50)
        features = [feature for feature, _ in shap_importances]

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
    )
    return prox_model


def prediction_confidence(predict_df: pd.DataFrame, prox_df: pd.DataFrame, id_column: str, target_column: str) -> None:
    """
    For each group in prox_df (grouped by id_column), compute the mean and stddev of target_column,
    merge with predict_df, and compare the 'prediction' with the computed mean.

    Confidence is assigned as:
      - "High" if prediction is within 1 std,
      - "Medium" if within 2 stds,
      - "Low" otherwise.

    Args:
        predict_df (pd.DataFrame): DataFrame with a 'prediction' column.
        prox_df (pd.DataFrame): DataFrame with neighbor info.
        id_column (str): Column name to group by (must be in both DataFrames).
        target_column (str): Column in prox_df to compute stats on.
    """
    # Group prox_df by id_column and compute mean and std for target_column
    stats_df = prox_df.groupby(id_column)[target_column].agg(["mean", "std"]).reset_index()

    # Merge stats with predict_df
    merged = predict_df.merge(stats_df, on=id_column, how="left")

    # Function to determine confidence based on prediction vs mean and std
    def compute_confidence(pred, mean, std):
        if pd.isna(std) or std == 0:
            return "Undefined" if pred != mean else "High"
        diff = abs(pred - mean)
        if diff <= std:
            return "High"
        elif diff <= 2 * std:
            return "Medium"
        else:
            return "Low"

    merged["confidence"] = merged.apply(
        lambda row: compute_confidence(row["prediction"], row["mean"], row["std"]), axis=1
    )

    # Print each group for inspection
    for group_id, group in merged.groupby(id_column):
        print(f"Group for {id_column} = {group_id}:")
        print(group[[id_column, "prediction", "mean", "std", "confidence"]])
        print("\n")


def try_load_xgboost_model(model_path: str):
    """Helper function to try loading an XGBoost model from a path."""
    import xgboost as xgb

    if os.path.exists(model_path):
        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
            return booster
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
    return None


def xgboost_model_from_s3(model_artifact_uri):
    """
    Download and extract XGBoost model artifact from S3, then load the model into memory.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact.

    Returns:
        Loaded XGBoost model or None if unavailable.
    """
    import xgboost as xgb

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model artifact
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

        # Extract tarball
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)

        # Start with common model paths
        possible_paths = [
            os.path.join(tmpdir, "xgboost-model"),
            os.path.join(tmpdir, "model"),
            os.path.join(tmpdir, "model.bin"),
        ]

        # Find all JSON model files and add to possible_paths
        possible_paths.extend(
            [
                os.path.join(root, file)
                for root, _, files in os.walk(tmpdir)
                for file in files
                if "model" in file.lower() and file.endswith(".json")
            ]
        )

        # Try each path
        for path in possible_paths:
            model = try_load_xgboost_model(path)
            if model:
                return model

        # If no XGBoost model found, look for pickled models
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".pkl") or file.endswith(".pickle"):
                    try:
                        import pickle

                        model_path = os.path.join(root, file)
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        if isinstance(model, xgb.Booster) or hasattr(model, "get_booster"):
                            # Return the booster if it's a pipeline with XGBoost
                            if hasattr(model, "get_booster"):
                                return model.get_booster()
                            return model
                    except Exception as e:
                        print(f"Failed to load pickled model from {file}: {e}")

    # If no model found
    return None


def feature_importance(workbench_model, importance_type: str = "weight") -> Optional[List[Tuple[str, float]]]:
    """
    Get sorted feature importances from an Workbench Model object.

    Args:
        workbench_model: Workbench model object
        importance_type: Type of feature importance.
            Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'

    Returns:
        List of tuples (feature, importance) sorted by importance value (descending)
        or None if there was an error
    """
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None

    # Get feature importances
    importances = xgb_model.get_score(importance_type=importance_type)

    # Convert to sorted list of tuples (feature, importance)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances


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

    # Prediction Confidence Testing
    # fmt: off
    prox_df = pd.DataFrame({
        "my_id": [
            "1", "1", "1", "1", "1",
            "2", "2", "2", "2", "2",
            "3", "3", "3", "3", "3",
            "4", "4", "4", "4", "4",
            "5", "5", "5", "5", "5"
        ],
        "neighbor_id": [
            "1", "2", "3", "4", "5",
            "2", "3", "4", "5", "6",
            "3", "4", "5", "6", "7",
            "4", "5", "6", "7", "8",
            "5", "6", "7", "8", "9"
        ],
        "distance": [
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.0, 0.1, 0.2, 0.3, 0.4
        ],
        "target": [
            1.0, 1.1, 1.2, 1.3, 1.4,
            2.0, 2.1, 2.2, 2.3, 2.4,
            3.0, 3.1, 3.2, 3.3, 3.4,
            4.0, 4.1, 4.2, 4.3, 4.4,
            5.0, 5.1, 5.2, 5.3, 5.4
        ]
    })
    # fmt: on

    predict_data = {
        "my_id": ["1", "2", "3", "4", "5"],
        "prediction": [1.1, 2.1, 3.1, 4.1, 5.1],
    }

    predict_df = pd.DataFrame(predict_data)

    # Call the prediction confidence function
    prediction_confidence(predict_df, prox_df, "my_id", "target")

    # Test the XGBoost model loading and feature importances
    model = Model("abalone-regression")
    feature_importance = feature_importance(model)
    print("Feature Importances:")
    print(feature_importance)
