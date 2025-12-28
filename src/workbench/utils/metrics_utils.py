"""Metrics utilities for computing model performance from predictions."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

log = logging.getLogger("workbench")


def validate_proba_columns(predictions_df: pd.DataFrame, class_labels: List[str], guessing: bool = False) -> bool:
    """Validate that probability columns match class labels.

    Args:
        predictions_df: DataFrame with prediction results
        class_labels: List of class labels
        guessing: Whether class labels were guessed from data

    Returns:
        True if validation passes

    Raises:
        ValueError: If probability columns don't match class labels
    """
    proba_columns = [col.replace("_proba", "") for col in predictions_df.columns if col.endswith("_proba")]

    if sorted(class_labels) != sorted(proba_columns):
        label_type = "GUESSED class_labels" if guessing else "class_labels"
        raise ValueError(f"_proba columns {proba_columns} != {label_type} {class_labels}!")
    return True


def compute_classification_metrics(
    predictions_df: pd.DataFrame,
    target_col: str,
    class_labels: Optional[List[str]] = None,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute classification metrics from a predictions DataFrame.

    Args:
        predictions_df: DataFrame with target and prediction columns
        target_col: Name of the target column
        class_labels: List of class labels in order (if None, inferred from target column)
        prediction_col: Name of the prediction column (default: "prediction")

    Returns:
        DataFrame with per-class metrics (precision, recall, f1, roc_auc, support)
        plus a weighted 'all' row. Returns empty DataFrame if validation fails.
    """
    # Validate inputs
    if predictions_df.empty:
        log.warning("Empty DataFrame provided. Returning empty metrics.")
        return pd.DataFrame()

    if prediction_col not in predictions_df.columns:
        log.warning(f"Prediction column '{prediction_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    if target_col not in predictions_df.columns:
        log.warning(f"Target column '{target_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    # Handle NaN predictions
    df = predictions_df.copy()
    nan_pred = df[prediction_col].isnull().sum()
    if nan_pred > 0:
        log.warning(f"Dropping {nan_pred} rows with NaN predictions.")
        df = df[~df[prediction_col].isnull()]

    if df.empty:
        log.warning("No valid rows after dropping NaNs. Returning empty metrics.")
        return pd.DataFrame()

    # Handle class labels
    guessing = False
    if class_labels is None:
        log.warning("Class labels not provided. Inferring from target column.")
        class_labels = df[target_col].unique().tolist()
        guessing = True

    # Validate probability columns if present
    proba_cols = [col for col in df.columns if col.endswith("_proba")]
    if proba_cols:
        validate_proba_columns(df, class_labels, guessing=guessing)

    y_true = df[target_col]
    y_pred = df[prediction_col]

    # Precision, recall, f1, support per class
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_labels, zero_division=0)

    # ROC AUC per class (requires probability columns and sorted labels)
    proba_col_names = [f"{label}_proba" for label in class_labels]
    if all(col in df.columns for col in proba_col_names):
        # roc_auc_score requires labels to be sorted, so we sort and reorder results back
        sorted_labels = sorted(class_labels)
        sorted_proba_cols = [f"{label}_proba" for label in sorted_labels]
        y_score_sorted = df[sorted_proba_cols].values
        roc_auc_sorted = roc_auc_score(y_true, y_score_sorted, labels=sorted_labels, multi_class="ovr", average=None)
        # Map back to original class_labels order
        label_to_auc = dict(zip(sorted_labels, roc_auc_sorted))
        roc_auc = np.array([label_to_auc[label] for label in class_labels])
    else:
        roc_auc = np.array([None] * len(class_labels))

    # Build per-class metrics
    metrics_df = pd.DataFrame(
        {
            target_col: class_labels,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "support": support.astype(int),
        }
    )

    # Add weighted 'all' row
    total = support.sum()
    all_row = {
        target_col: "all",
        "precision": (prec * support).sum() / total,
        "recall": (rec * support).sum() / total,
        "f1": (f1 * support).sum() / total,
        "roc_auc": (roc_auc * support).sum() / total if roc_auc[0] is not None else None,
        "support": int(total),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([all_row])], ignore_index=True)

    return metrics_df


def compute_regression_metrics(
    predictions_df: pd.DataFrame,
    target_col: str,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute regression metrics from a predictions DataFrame.

    Args:
        predictions_df: DataFrame with target and prediction columns
        target_col: Name of the target column
        prediction_col: Name of the prediction column (default: "prediction")

    Returns:
        DataFrame with regression metrics (rmse, mae, medae, r2, spearmanr, support)
        Returns empty DataFrame if validation fails or no valid data.
    """
    # Validate inputs
    if predictions_df.empty:
        log.warning("Empty DataFrame provided. Returning empty metrics.")
        return pd.DataFrame()

    if prediction_col not in predictions_df.columns:
        log.warning(f"Prediction column '{prediction_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    if target_col not in predictions_df.columns:
        log.warning(f"Target column '{target_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    # Handle NaN values
    df = predictions_df[[target_col, prediction_col]].copy()
    nan_target = df[target_col].isnull().sum()
    nan_pred = df[prediction_col].isnull().sum()
    if nan_target > 0 or nan_pred > 0:
        log.warning(f"NaNs found: {target_col}={nan_target}, {prediction_col}={nan_pred}. Dropping NaN rows.")
        df = df.dropna()

    if df.empty:
        log.warning("No valid rows after dropping NaNs. Returning empty metrics.")
        return pd.DataFrame()

    y_true = df[target_col].values
    y_pred = df[prediction_col].values

    return pd.DataFrame(
        [
            {
                "rmse": root_mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "medae": median_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "spearmanr": spearmanr(y_true, y_pred).correlation,
                "support": len(y_true),
            }
        ]
    )


def compute_metrics_from_predictions(
    predictions_df: pd.DataFrame,
    target_col: str,
    class_labels: Optional[List[str]] = None,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute metrics from a predictions DataFrame.

    Automatically determines if this is classification or regression based on
    whether class_labels is provided.

    Args:
        predictions_df: DataFrame with target and prediction columns
        target_col: Name of the target column
        class_labels: List of class labels for classification (None for regression)
        prediction_col: Name of the prediction column (default: "prediction")

    Returns:
        DataFrame with computed metrics
    """
    if target_col not in predictions_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in predictions DataFrame")
    if prediction_col not in predictions_df.columns:
        raise ValueError(f"Prediction column '{prediction_col}' not found in predictions DataFrame")

    if class_labels:
        return compute_classification_metrics(predictions_df, target_col, class_labels, prediction_col)
    else:
        return compute_regression_metrics(predictions_df, target_col, prediction_col)


if __name__ == "__main__":
    # Test with sample data
    print("Testing classification metrics...")
    class_df = pd.DataFrame(
        {
            "target": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "prediction": ["a", "b", "c", "a", "b", "a", "a", "b", "c", "b"],
            "a_proba": [0.8, 0.1, 0.1, 0.7, 0.2, 0.4, 0.9, 0.1, 0.1, 0.3],
            "b_proba": [0.1, 0.8, 0.1, 0.2, 0.7, 0.3, 0.05, 0.8, 0.2, 0.6],
            "c_proba": [0.1, 0.1, 0.8, 0.1, 0.1, 0.3, 0.05, 0.1, 0.7, 0.1],
        }
    )
    metrics = compute_metrics_from_predictions(class_df, "target", ["a", "b", "c"])
    print(metrics.to_string(index=False))

    print("\nTesting regression metrics...")
    reg_df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "prediction": [1.1, 2.2, 2.9, 4.1, 4.8],
        }
    )
    metrics = compute_metrics_from_predictions(reg_df, "target")
    print(metrics.to_string(index=False))
