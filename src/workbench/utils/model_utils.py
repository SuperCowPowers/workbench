"""Model Utilities for Workbench models"""

import logging
import pandas as pd
import numpy as np
import importlib.resources
from pathlib import Path
import os
import json
import tempfile
import tarfile
import awswrangler as wr
from typing import Optional, Dict, Any
from scipy.stats import norm

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
        model_type=ModelType.PROXIMITY,
        feature_list=features,
        target_column=target,
        description=f"Proximity Model for {model.uuid}",
        tags=["proximity", model.uuid],
        custom_script=script_path,
        custom_args={"track_columns": track_columns},
    )
    return prox_model


def uq_model(model: "Model", uq_model_name: str, train_all_data: bool = False) -> "Model":
    """Create a Uncertainty Quantification (UQ) model based on the given model

    Args:
        model (Model): The model to create the UQ model from
        uq_model_name (str): The name of the UQ model to create
        train_all_data (bool, optional): Whether to train the UQ model on all data (default: False)

    Returns:
        Model: The UQ model
    """
    from workbench.api import Model, ModelType, FeatureSet  # noqa: F401 (avoid circular import)

    # Get the custom script path for the UQ model
    script_path = get_custom_script_path("uq_models", "meta_uq.template")

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Create the Proximity Model from our FeatureSet
    fs = FeatureSet(model.get_input())
    uq_model = fs.to_model(
        name=uq_model_name,
        model_type=ModelType.UQ_REGRESSOR,
        feature_list=features,
        target_column=target,
        description=f"UQ Model for {model.uuid}",
        tags=["uq", model.uuid],
        train_all_data=train_all_data,
        custom_script=script_path,
        custom_args={"id_column": fs.id_column, "track_columns": [target]}
    )
    return uq_model


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


def uq_metrics(df: pd.DataFrame, target_col: str, model_name: Optional[str] = "UQ Model") -> Dict[str, Any]:
    """
    Evaluate uncertainty quantification model with standard metrics, including coverage,
    interval width, correlation between uncertainty and error, CRPS, and calibration error.
    Args:
        df: DataFrame with predictions and uncertainty estimates.
            Must contain the target column, a prediction column ("prediction"), and either
            quantile columns ("q_025", "q_975", "q_25", "q_75") or a standard deviation
            column ("prediction_std").
        target_col: Name of the true target column in the DataFrame.
        model_name: Optional name of the model for reporting purposes.
    Returns:
        Dictionary of computed metrics.
    Raises:
        ValueError: If required columns are missing or if the input DataFrame is empty.
    """
    # Input Validation
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if "prediction" not in df.columns:
        raise ValueError("Prediction column 'prediction' not found in DataFrame.")

    # --- Basic Statistics ---
    residuals = df[target_col] - df["prediction"]
    abs_residuals = np.abs(residuals)
    rmse = np.sqrt(np.mean(residuals**2))

    # --- Uncertainty-Error Correlation ---
    if "prediction_std" in df.columns:
        uncertainty_error_corr = np.corrcoef(df["prediction_std"], abs_residuals)[0, 1]
    else:
        uncertainty_error_corr = np.nan  # or skip this metric

    # --- Coverage and Interval Width ---
    if "q_025" in df.columns and "q_975" in df.columns:
        lower_95, upper_95 = df["q_025"], df["q_975"]
        lower_50, upper_50 = df["q_25"], df["q_75"]
    elif "prediction_std" in df.columns:
        # Using standard deviation to define intervals
        lower_95 = df["prediction"] - 1.96 * df["prediction_std"]
        upper_95 = df["prediction"] + 1.96 * df["prediction_std"]
        lower_50 = df["prediction"] - 0.674 * df["prediction_std"]
        upper_50 = df["prediction"] + 0.674 * df["prediction_std"]
    else:
        raise ValueError(
            "Either quantile columns (q_025, q_975, q_25, q_75) or 'prediction_std' column must be present."
        )
    coverage_95 = np.mean((df[target_col] >= lower_95) & (df[target_col] <= upper_95))
    coverage_50 = np.mean((df[target_col] >= lower_50) & (df[target_col] <= upper_50))
    avg_width_95 = np.mean(upper_95 - lower_95)
    avg_width_50 = np.mean(upper_50 - lower_50)

    # --- Interval Score ---
    alpha_95, alpha_50 = 0.05, 0.50
    is_95 = (
        (upper_95 - lower_95)
        + (2 / alpha_95) * (lower_95 - df[target_col]) * (df[target_col] < lower_95)
        + (2 / alpha_95) * (df[target_col] - upper_95) * (df[target_col] > upper_95)
    )
    mean_is_95 = np.mean(is_95)
    is_50 = (
        (upper_50 - lower_50)
        + (2 / alpha_50) * (lower_50 - df[target_col]) * (df[target_col] < lower_50)
        + (2 / alpha_50) * (df[target_col] - upper_50) * (df[target_col] > upper_50)
    )
    mean_is_50 = np.mean(is_50)

    # --- Negative Log-Likelihood (NLL) ---
    if "prediction_std" in df.columns:
        nll = 0.5 * np.log(2 * np.pi) + np.log(df["prediction_std"]) + (residuals**2) / (2 * df["prediction_std"] ** 2)
        mean_nll = np.mean(nll)
    else:
        mean_nll = np.nan

    # --- Continuous Ranked Probability Score (CRPS) ---
    if "prediction_std" in df.columns:
        crps = (
            norm.cdf((df[target_col] - df["prediction"]) / df["prediction_std"])
            - (df[target_col] - df["prediction"]) / df["prediction_std"]
        )
        mean_crps = np.mean(crps)
    else:
        mean_crps = np.nan

    # --- Calibration Error ---
    # Group predictions by decile and check if the observed frequency matches the expected
    df["quantile"] = pd.qcut(df["prediction"], q=10, labels=False)  # Assign deciles
    calibration_error = np.mean(
        np.abs(df.groupby("quantile")[target_col].mean() - df.groupby("quantile")["prediction"].mean())
    )
    results = {
        "model": model_name,
        "coverage_95": coverage_95,
        "coverage_50": coverage_50,
        "avg_width_95": avg_width_95,
        "avg_width_50": avg_width_50,
        "uncertainty_correlation": uncertainty_error_corr,
        "negative_log_likelihood": mean_nll,
        "interval_score_95": mean_is_95,
        "interval_score_50": mean_is_50,
        "rmse": rmse,
        "crps": mean_crps,
        "calibration_error": calibration_error,
        "n_samples": len(df),
    }
    print(f"\n=== {model_name} UQ Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"Coverage @ 95%: {coverage_95:.3f} (target: 0.95)")
    print(f"Coverage @ 50%: {coverage_50:.3f} (target: 0.50)")
    print(f"Average 95% Width: {avg_width_95:.3f}")
    print(f"Average 50% Width: {avg_width_50:.3f}")
    if "prediction_std" in df.columns:
        print(f"Uncertainty-Error Correlation: {uncertainty_error_corr:.3f}")
        print(f"Negative Log-Likelihood: {mean_nll:.3f}")
        print(f"CRPS: {mean_crps:.3f}")
    print(f"Interval Score 95%: {mean_is_95:.3f}")
    print(f"Interval Score 50%: {mean_is_50:.3f}")
    print(f"Calibration Error: {calibration_error:.3f}")
    print(f"Samples: {len(df)}")
    return results


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from pprint import pprint
    from workbench.api import Model, Endpoint

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
    m = Model("aqsol-regression")
    # prox_model = proximity_model(m, "aqsol-prox")
    # print(prox_model)#

    # Test the UQ model
    # uq_model_instance = uq_model(m, "aqsol-uq")
    # print(uq_model_instance)
    # uq_model_instance.to_endpoint()

    # Test the uq_metrics function
    end = Endpoint("aqsol-uq")
    df = end.auto_inference()
    results = uq_metrics(df, target_col="solubility")
    pprint(results)
