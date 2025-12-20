"""ChemProp utilities for Workbench models."""

import logging
import os
from typing import Any, Tuple

import pandas as pd

from workbench.utils.aws_utils import pull_s3_data
from workbench.utils.metrics_utils import compute_metrics_from_predictions
from workbench.utils.model_utils import safe_extract_tarfile

log = logging.getLogger("workbench")


def download_and_extract_model(s3_uri: str, model_dir: str) -> None:
    """Download model artifact from S3 and extract it.

    Args:
        s3_uri: S3 URI to the model artifact (model.tar.gz)
        model_dir: Directory to extract model artifacts to
    """
    import awswrangler as wr

    log.info(f"Downloading model from {s3_uri}...")

    # Download to temp file
    local_tar_path = os.path.join(model_dir, "model.tar.gz")
    wr.s3.download(path=s3_uri, local_file=local_tar_path)

    # Extract using safe extraction
    log.info(f"Extracting to {model_dir}...")
    safe_extract_tarfile(local_tar_path, model_dir)

    # Cleanup tar file
    os.unlink(local_tar_path)


def load_chemprop_model_artifacts(model_dir: str) -> Tuple[Any, dict]:
    """Load ChemProp MPNN model and artifacts from an extracted model directory.

    Args:
        model_dir: Directory containing extracted model artifacts

    Returns:
        Tuple of (MPNN model, artifacts_dict).
        artifacts_dict contains 'label_encoder' and 'feature_metadata' if present.
    """
    import joblib
    from chemprop import models

    model_path = os.path.join(model_dir, "chemprop_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No chemprop_model.pt found in {model_dir}")

    model = models.MPNN.load_from_file(model_path)
    model.eval()

    # Load additional artifacts
    artifacts = {}

    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        artifacts["label_encoder"] = joblib.load(label_encoder_path)

    feature_metadata_path = os.path.join(model_dir, "feature_metadata.joblib")
    if os.path.exists(feature_metadata_path):
        artifacts["feature_metadata"] = joblib.load(feature_metadata_path)

    return model, artifacts


def pull_cv_results(workbench_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cross-validation results from AWS training artifacts.

    This retrieves the validation predictions saved during model training and
    computes metrics directly from them.

    Args:
        workbench_model: Workbench model object

    Returns:
        Tuple of:
            - DataFrame with computed metrics
            - DataFrame with validation predictions
    """
    # Get the validation predictions from S3
    s3_path = f"{workbench_model.model_training_path}/validation_predictions.csv"
    predictions_df = pull_s3_data(s3_path)

    if predictions_df is None:
        raise ValueError(f"No validation predictions found at {s3_path}")

    log.info(f"Pulled {len(predictions_df)} validation predictions from {s3_path}")

    # Compute metrics from predictions
    target = workbench_model.target()
    class_labels = workbench_model.class_labels()

    if target in predictions_df.columns and "prediction" in predictions_df.columns:
        metrics_df = compute_metrics_from_predictions(predictions_df, target, class_labels)
    else:
        metrics_df = pd.DataFrame()

    return metrics_df, predictions_df


if __name__ == "__main__":

    # Tests for the ChemProp utilities
    from workbench.api import Model

    # Initialize Workbench model
    model_name = "logd-reg-chemprop"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Pull CV results
    metrics_df, predictions_df = pull_cv_results(model)
    print("\nTraining Metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSample Predictions:\n{predictions_df.head().to_string(index=False)}")
