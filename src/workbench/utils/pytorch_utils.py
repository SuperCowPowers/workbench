"""PyTorch Tabular utilities for Workbench models."""

import logging
import os
import tarfile
import tempfile
from typing import Any, Tuple

import awswrangler as wr
import pandas as pd

from workbench.utils.aws_utils import pull_s3_data
from workbench.utils.metrics_utils import compute_metrics_from_predictions

log = logging.getLogger("workbench")


def download_and_extract_model(s3_uri: str, model_dir: str) -> None:
    """Download and extract a PyTorch model artifact from S3.

    Args:
        s3_uri: S3 URI of the model.tar.gz artifact
        model_dir: Local directory to extract the model to
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wr.s3.download(path=s3_uri, local_file=tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(model_dir)
        log.info(f"Extracted model to {model_dir}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def pull_cv_results(workbench_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cross-validation results from AWS training artifacts.

    This retrieves the validation predictions saved during model training and
    computes metrics directly from them. For PyTorch models trained with
    n_folds > 1, these are out-of-fold predictions from k-fold cross-validation.

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
    from workbench.api import Model

    # Test pulling CV results
    model_name = "aqsol-reg-pytorch"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Pull CV results from training artifacts
    metrics_df, predictions_df = pull_cv_results(model)
    print(f"\nMetrics:\n{metrics_df}")
    print(f"\nPredictions shape: {predictions_df.shape}")
    print(f"Predictions columns: {predictions_df.columns.tolist()}")
