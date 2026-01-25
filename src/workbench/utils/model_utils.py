"""Model Utilities for Workbench models"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
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


def proximity_model_local(model: "Model", include_all_columns: bool = False):
    """Create a FeatureSpaceProximity Model for this Model

    Args:
        model (Model): The Model/FeatureSet used to create the proximity model
        include_all_columns (bool): Include all DataFrame columns in neighbor results (default: False)

    Returns:
        FeatureSpaceProximity: The proximity model
    """
    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity  # noqa: F401
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Create the Proximity Model from both the full FeatureSet and the Model training data
    full_df = fs.pull_dataframe()
    model_df = model.training_view().pull_dataframe()

    # Mark rows that are in the model
    model_ids = set(model_df[id_column])
    full_df["in_model"] = full_df[id_column].isin(model_ids)

    # Create and return the FeatureSpaceProximity Model
    return FeatureSpaceProximity(
        full_df, id_column=id_column, features=features, target=target, include_all_columns=include_all_columns
    )


def fingerprint_prox_model_local(
    model: "Model",
    include_all_columns: bool = False,
    radius: int = 2,
    n_bits: int = 1024,
    counts: bool = False,
):
    """Create a FingerprintProximity Model for this Model

    Args:
        model (Model): The Model used to create the fingerprint proximity model
        include_all_columns (bool): Include all DataFrame columns in neighbor results (default: False)
        radius (int): Morgan fingerprint radius (default: 2)
        n_bits (int): Number of bits for the fingerprint (default: 1024)
        counts (bool): Use count fingerprints instead of binary (default: False)

    Returns:
        FingerprintProximity: The fingerprint proximity model
    """
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity  # noqa: F401
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Target Column from the existing given Model
    target = model.target()

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Create the Proximity Model from both the full FeatureSet and the Model training data
    full_df = fs.pull_dataframe()
    model_df = model.training_view().pull_dataframe()

    # Mark rows that are in the model
    model_ids = set(model_df[id_column])
    full_df["in_model"] = full_df[id_column].isin(model_ids)

    # Create and return the FingerprintProximity Model
    return FingerprintProximity(
        full_df,
        id_column=id_column,
        target=target,
        include_all_columns=include_all_columns,
        radius=radius,
        n_bits=n_bits,
    )


def noise_model_local(model: "Model"):
    """Create a NoiseModel for detecting noisy/problematic samples in a Model's training data.

    Args:
        model (Model): The Model used to create the noise model

    Returns:
        NoiseModel: The noise model with precomputed noise scores for all samples
    """
    from workbench.algorithms.models.noise_model import NoiseModel  # noqa: F401 (avoid circular import)
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Create the NoiseModel from both the full FeatureSet and the Model training data
    full_df = fs.pull_dataframe()
    model_df = model.training_view().pull_dataframe()

    # Mark rows that are in the model
    model_ids = set(model_df[id_column])
    full_df["in_model"] = full_df[id_column].isin(model_ids)

    # Create and return the NoiseModel
    return NoiseModel(full_df, id_column, features, target)


def cleanlab_model_local(model: "Model"):
    """Create a CleanlabModels instance for detecting data quality issues in a Model's training data.

    Args:
        model (Model): The Model used to create the cleanlab models

    Returns:
        CleanlabModels: Factory providing access to CleanLearning and Datalab models.
            - clean_learning(): CleanLearning model with enhanced get_label_issues()
            - datalab(): Datalab instance with report(), get_issues()
    """
    from workbench.algorithms.models.cleanlab_model import create_cleanlab_model  # noqa: F401 (avoid circular import)
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()
    model_type = model.model_type

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Get the full FeatureSet data
    full_df = fs.pull_dataframe()

    # Create and return the CleanLearning model
    return create_cleanlab_model(full_df, id_column, features, target, model_type=model_type)


def published_proximity_model(model: "Model", prox_model_name: str, include_all_columns: bool = False) -> "Model":
    """Create a published proximity model based on the given model

    Args:
        model (Model): The model to create the proximity model from
        prox_model_name (str): The name of the proximity model to create
        include_all_columns (bool): Include all DataFrame columns in results (default: False)
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
        description=f"Proximity Model for {model.name}",
        tags=["proximity", model.name],
        custom_script=script_path,
        custom_args={"include_all_columns": include_all_columns},
    )
    return prox_model


def safe_extract_tarfile(tar_path: str, extract_path: str) -> None:
    """
    Extract a tarball safely, using data filter if available.

    The filter parameter was backported to Python 3.8+, 3.9+, 3.10.13+, 3.11+
    as a security patch, but may not be present in older patch versions.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=extract_path, filter="data")
        else:
            tar.extractall(path=extract_path)


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
        safe_extract_tarfile(local_tar_path, tmpdir)

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


def load_hyperparameters_from_s3(model_artifact_uri: str) -> Optional[dict]:
    """
    Download and extract hyperparameters from a model artifact in S3.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact (model.tar.gz).

    Returns:
        dict: The loaded hyperparameters or None if not found.
    """
    hyperparameters = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model artifact
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

        # Extract tarball
        safe_extract_tarfile(local_tar_path, tmpdir)

        # Look for hyperparameters in base directory only
        hyperparameters_path = os.path.join(tmpdir, "hyperparameters.json")

        if os.path.exists(hyperparameters_path):
            try:
                with open(hyperparameters_path, "r") as f:
                    hyperparameters = json.load(f)
                log.info(f"Loaded hyperparameters from {hyperparameters_path}")
            except Exception as e:
                log.warning(f"Failed to load hyperparameters from {hyperparameters_path}: {e}")

    return hyperparameters


def get_model_hyperparameters(workbench_model: Any) -> Optional[dict]:
    """Get the hyperparameters used to train a Workbench model.

    This retrieves the hyperparameters.json file from the model artifacts
    that was saved during model training.

    Args:
        workbench_model: Workbench model object

    Returns:
        dict: The hyperparameters used during training, or None if not found
    """
    # Get the model artifact URI
    model_artifact_uri = workbench_model.model_data_url()

    if model_artifact_uri is None:
        log.warning(f"No model artifact found for {workbench_model.uuid}")
        return None

    log.info(f"Loading hyperparameters from {model_artifact_uri}")
    return load_hyperparameters_from_s3(model_artifact_uri)


def uq_metrics(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Evaluate uncertainty quantification model with essential metrics.
    Args:
        df: DataFrame with predictions and uncertainty estimates.
            Must contain the target column, a prediction column ("prediction"), and either
            quantile columns ("q_025", "q_975", "q_25", "q_75") or a standard deviation
            column ("prediction_std").
        target_col: Name of the true target column in the DataFrame.
    Returns:
        Dictionary of computed metrics.
    """
    # Input Validation
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if "prediction" not in df.columns:
        raise ValueError("Prediction column 'prediction' not found in DataFrame.")

    # Drop rows with NaN predictions (e.g., from models that can't handle missing features)
    n_total = len(df)
    df = df.dropna(subset=["prediction", target_col])
    n_valid = len(df)
    if n_valid < n_total:
        log.info(f"UQ metrics: dropped {n_total - n_valid} rows with NaN predictions")

    # --- Coverage and Interval Width ---
    if "q_025" in df.columns and "q_975" in df.columns:
        lower_95, upper_95 = df["q_025"], df["q_975"]
        lower_90, upper_90 = df["q_05"], df["q_95"]
        lower_80, upper_80 = df["q_10"], df["q_90"]
        lower_68 = df.get("q_16", df["q_10"])  # fallback to 80% interval
        upper_68 = df.get("q_84", df["q_90"])  # fallback to 80% interval
        lower_50, upper_50 = df["q_25"], df["q_75"]
    elif "prediction_std" in df.columns:
        lower_95 = df["prediction"] - 1.96 * df["prediction_std"]
        upper_95 = df["prediction"] + 1.96 * df["prediction_std"]
        lower_90 = df["prediction"] - 1.645 * df["prediction_std"]
        upper_90 = df["prediction"] + 1.645 * df["prediction_std"]
        lower_80 = df["prediction"] - 1.282 * df["prediction_std"]
        upper_80 = df["prediction"] + 1.282 * df["prediction_std"]
        lower_68 = df["prediction"] - 1.0 * df["prediction_std"]
        upper_68 = df["prediction"] + 1.0 * df["prediction_std"]
        lower_50 = df["prediction"] - 0.674 * df["prediction_std"]
        upper_50 = df["prediction"] + 0.674 * df["prediction_std"]
    else:
        raise ValueError(
            "Either quantile columns (q_025, q_975, q_25, q_75) or 'prediction_std' column must be present."
        )
    median_std = df["prediction_std"].median()
    coverage_95 = np.mean((df[target_col] >= lower_95) & (df[target_col] <= upper_95))
    coverage_90 = np.mean((df[target_col] >= lower_90) & (df[target_col] <= upper_90))
    coverage_80 = np.mean((df[target_col] >= lower_80) & (df[target_col] <= upper_80))
    coverage_68 = np.mean((df[target_col] >= lower_68) & (df[target_col] <= upper_68))
    median_width_95 = np.median(upper_95 - lower_95)
    median_width_90 = np.median(upper_90 - lower_90)
    median_width_80 = np.median(upper_80 - lower_80)
    median_width_50 = np.median(upper_50 - lower_50)
    median_width_68 = np.median(upper_68 - lower_68)

    # --- CRPS (measures calibration + sharpness) ---
    z = (df[target_col] - df["prediction"]) / df["prediction_std"]
    crps = df["prediction_std"] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    mean_crps = np.mean(crps)

    # --- Interval Score @ 95% (penalizes miscoverage) ---
    alpha_95 = 0.05
    is_95 = (
        (upper_95 - lower_95)
        + (2 / alpha_95) * (lower_95 - df[target_col]) * (df[target_col] < lower_95)
        + (2 / alpha_95) * (df[target_col] - upper_95) * (df[target_col] > upper_95)
    )
    mean_is_95 = np.mean(is_95)

    # --- Interval to Error Correlation ---
    abs_residuals = np.abs(df[target_col] - df["prediction"])
    width_68 = upper_68 - lower_68

    # Spearman correlation for robustness
    interval_to_error_corr = spearmanr(width_68, abs_residuals)[0]

    # --- Confidence to Error Correlation ---
    # If confidence column exists, compute correlation (should be negative: high confidence = low error)
    confidence_to_error_corr = None
    if "confidence" in df.columns:
        confidence_to_error_corr = spearmanr(df["confidence"], abs_residuals)[0]

    # Collect results
    results = {
        "coverage_68": coverage_68,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "median_std": median_std,
        "median_width_50": median_width_50,
        "median_width_68": median_width_68,
        "median_width_80": median_width_80,
        "median_width_90": median_width_90,
        "median_width_95": median_width_95,
        "interval_to_error_corr": interval_to_error_corr,
        "confidence_to_error_corr": confidence_to_error_corr,
        "n_samples": len(df),
    }

    print("\n=== UQ Metrics ===")
    print(f"Coverage @ 68%: {coverage_68:.3f} (target: 0.68)")
    print(f"Coverage @ 80%: {coverage_80:.3f} (target: 0.80)")
    print(f"Coverage @ 90%: {coverage_90:.3f} (target: 0.90)")
    print(f"Coverage @ 95%: {coverage_95:.3f} (target: 0.95)")
    print(f"Median Prediction StdDev: {median_std:.3f}")
    print(f"Median 50% Width: {median_width_50:.3f}")
    print(f"Median 68% Width: {median_width_68:.3f}")
    print(f"Median 80% Width: {median_width_80:.3f}")
    print(f"Median 90% Width: {median_width_90:.3f}")
    print(f"Median 95% Width: {median_width_95:.3f}")
    print(f"CRPS: {mean_crps:.3f} (lower is better)")
    print(f"Interval Score 95%: {mean_is_95:.3f} (lower is better)")
    print(f"Interval/Error Corr: {interval_to_error_corr:.3f} (higher is better, target: >0.5)")
    if confidence_to_error_corr is not None:
        print(f"Confidence/Error Corr: {confidence_to_error_corr:.3f} (lower is better, target: <-0.5)")
    print(f"Samples: {len(df)}")
    return results


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

    # Test loading hyperparameters
    m = Model("aqsol-regression")
    hyperparams = get_model_hyperparameters(m)
    print(hyperparams)

    # Test the proximity model
    # prox_model = proximity_model(m, "aqsol-prox")
    # print(prox_model)#
