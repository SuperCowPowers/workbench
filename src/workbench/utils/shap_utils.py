"""SHAP Utilities for retrieving SHAP values from S3 training artifacts

For models trained with recent templates, SHAP values are computed during training
and stored in S3 alongside other model artifacts. This module provides functions
to retrieve those pre-computed SHAP values.
"""

import logging
from typing import Optional, List, Tuple, Dict, Union

import pandas as pd

# Workbench Imports
from workbench.utils.aws_utils import pull_s3_data

# Set up the log
log = logging.getLogger("workbench")


def get_shap_importance(model_training_path: str) -> Optional[List[Tuple[str, float]]]:
    """Retrieve SHAP feature importance from S3 training artifacts.

    Args:
        model_training_path: S3 path to model training artifacts (e.g., s3://bucket/models/my-model/)

    Returns:
        List of tuples (feature_name, importance_score) sorted by importance descending,
        or None if not found
    """
    if model_training_path is None:
        return None

    s3_path = f"{model_training_path}/shap_importance.json"
    try:
        import awswrangler as wr

        data = wr.s3.read_json(s3_path)
        # Convert from list of lists to list of tuples
        return [tuple(item) for item in data]
    except Exception:
        return None


def get_shap_values(
    model_training_path: str, class_labels: List[str] = None
) -> Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Retrieve SHAP values from S3 training artifacts.

    Args:
        model_training_path: S3 path to model training artifacts
        class_labels: List of class labels for multiclass models (optional)

    Returns:
        For regression/binary: DataFrame with SHAP values
        For multiclass: Dict mapping class label to DataFrame
        Returns None if not found
    """
    if model_training_path is None:
        return None

    # Try single file first (regression/binary classification)
    s3_path = f"{model_training_path}/shap_values.csv"
    df = pull_s3_data(s3_path)
    if df is not None:
        return df

    # Try multiclass files if class_labels provided
    if class_labels:
        shap_data = {}
        for label in class_labels:
            s3_path = f"{model_training_path}/shap_values_{label}.csv"
            df = pull_s3_data(s3_path)
            if df is not None:
                shap_data[label] = df
        return shap_data if shap_data else None

    return None


def get_shap_feature_values(model_training_path: str) -> Optional[pd.DataFrame]:
    """Retrieve SHAP feature values (sample data for plotting) from S3.

    Args:
        model_training_path: S3 path to model training artifacts

    Returns:
        DataFrame with feature values for the SHAP sample rows, or None if not found
    """
    if model_training_path is None:
        return None

    s3_path = f"{model_training_path}/shap_feature_values.csv"
    return pull_s3_data(s3_path)


if __name__ == "__main__":
    """Exercise the SHAP Utilities"""
    from workbench.api import Model

    # Test with a model
    model = Model("abalone-regression")
    training_path = model.model_training_path

    print("\n=== SHAP Importance ===")
    importance = get_shap_importance(training_path)
    if importance:
        for feature, score in importance[:5]:
            print(f"  {feature}: {score:.4f}")

    print("\n=== SHAP Values ===")
    shap_values = get_shap_values(training_path)
    if shap_values is not None:
        if isinstance(shap_values, dict):
            for label, df in shap_values.items():
                print(f"  Class {label}: {df.shape}")
        else:
            print(f"  Shape: {shap_values.shape}")

    print("\n=== SHAP Feature Values ===")
    feature_values = get_shap_feature_values(training_path)
    if feature_values is not None:
        print(f"  Shape: {feature_values.shape}")
