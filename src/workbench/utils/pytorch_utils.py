"""PyTorch Tabular utilities for Workbench models."""

import logging
from typing import Any, Tuple

import pandas as pd

from workbench.utils.aws_utils import pull_s3_data

log = logging.getLogger("workbench")


def pull_cv_results(workbench_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cross-validation results from AWS training artifacts.

    This retrieves the validation predictions and training metrics that were
    saved during model training. For PyTorch models trained with n_folds > 1,
    these are out-of-fold predictions from k-fold cross-validation.

    Args:
        workbench_model: Workbench model object

    Returns:
        Tuple of:
            - DataFrame with training metrics
            - DataFrame with validation predictions
    """
    # Get the validation predictions from S3
    s3_path = f"{workbench_model.model_training_path}/validation_predictions.csv"
    predictions_df = pull_s3_data(s3_path)

    if predictions_df is None:
        raise ValueError(f"No validation predictions found at {s3_path}")

    log.info(f"Pulled {len(predictions_df)} validation predictions from {s3_path}")

    # Get training metrics from model metadata
    training_metrics = workbench_model.workbench_meta().get("workbench_training_metrics")

    if training_metrics is None:
        log.warning(f"No training metrics found in model metadata for {workbench_model.model_name}")
        metrics_df = pd.DataFrame({"error": [f"No training metrics found for {workbench_model.model_name}"]})
    else:
        metrics_df = pd.DataFrame.from_dict(training_metrics)
        log.info(f"Metrics summary:\n{metrics_df.to_string(index=False)}")

    return metrics_df, predictions_df


if __name__ == "__main__":
    from workbench.api import Model

    # Test pulling CV results
    model_name = "aqsol-pytorch-reg"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Pull CV results from training artifacts
    metrics_df, predictions_df = pull_cv_results(model)
    print(f"\nMetrics:\n{metrics_df}")
    print(f"\nPredictions shape: {predictions_df.shape}")
    print(f"Predictions columns: {predictions_df.columns.tolist()}")
