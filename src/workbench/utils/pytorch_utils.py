"""PyTorch Tabular utilities for Workbench models."""

import logging
import os
import tarfile
import tempfile
from typing import Any, Tuple

import awswrangler as wr
import pandas as pd

from workbench.utils.metrics_utils import compute_metrics_from_predictions
from workbench.utils.model_utils import pull_oof_predictions

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

    This retrieves the out-of-fold predictions saved during model training and
    computes metrics directly from them. Each row is scored by the one fold model
    that held it out, so these are leak-free single-model predictions.

    Args:
        workbench_model: Workbench model object

    Returns:
        Tuple of:
            - DataFrame with computed metrics
            - DataFrame with out-of-fold predictions
    """
    # Get the out-of-fold predictions from the model's training artifacts
    predictions_df = pull_oof_predictions(workbench_model)

    if predictions_df is None:
        raise ValueError(f"No out-of-fold predictions for {workbench_model.name} (retrain the model to generate them)")

    if predictions_df.empty:
        log.warning(f"No out-of-fold predictions for {workbench_model.name}")
        return pd.DataFrame(), predictions_df

    log.info(f"Pulled {len(predictions_df)} out-of-fold predictions for {workbench_model.name}")

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
