"""Shared utility functions for model training scripts (templates).

These functions are used across multiple model templates (XGBoost, PyTorch, ChemProp)
to reduce code duplication and ensure consistent behavior.
"""

from io import StringIO
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)
from scipy.stats import spearmanr


def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    """Check if the provided dataframe is empty and raise an exception if it is.

    Args:
        df: DataFrame to check
        df_name: Name of the DataFrame (for error message)

    Raises:
        ValueError: If the DataFrame is empty
    """
    if df.empty:
        msg = f"*** The training data {df_name} has 0 rows! ***STOPPING***"
        print(msg)
        raise ValueError(msg)


def expand_proba_column(df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Expands a column containing a list of probabilities into separate columns.

    Handles None values for rows where predictions couldn't be made.

    Args:
        df: DataFrame containing a "pred_proba" column
        class_labels: List of class labels

    Returns:
        DataFrame with the "pred_proba" expanded into separate columns (e.g., "class1_proba")

    Raises:
        ValueError: If DataFrame does not contain a "pred_proba" column
    """
    proba_column = "pred_proba"
    if proba_column not in df.columns:
        raise ValueError('DataFrame does not contain a "pred_proba" column')

    proba_splits = [f"{label}_proba" for label in class_labels]
    n_classes = len(class_labels)

    # Handle None values by replacing with list of NaNs
    proba_values = []
    for val in df[proba_column]:
        if val is None:
            proba_values.append([np.nan] * n_classes)
        else:
            proba_values.append(val)

    proba_df = pd.DataFrame(proba_values, columns=proba_splits)

    # Drop any existing proba columns and reset index for concat
    df = df.drop(columns=[proba_column] + proba_splits, errors="ignore")
    df = df.reset_index(drop=True)
    df = pd.concat([df, proba_df], axis=1)
    return df


def match_features_case_insensitive(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    """Matches and renames DataFrame columns to match model feature names (case-insensitive).

    Prioritizes exact matches, then case-insensitive matches.

    Args:
        df: Input DataFrame
        model_features: List of feature names expected by the model

    Returns:
        DataFrame with columns renamed to match model features

    Raises:
        ValueError: If any model features cannot be matched
    """
    df_columns_lower = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing = []
    for feature in model_features:
        if feature in df.columns:
            continue  # Exact match
        elif feature.lower() in df_columns_lower:
            rename_dict[df_columns_lower[feature.lower()]] = feature
        else:
            missing.append(feature)

    if missing:
        raise ValueError(f"Features not found: {missing}")

    return df.rename(columns=rename_dict)


def convert_categorical_types(
    df: pd.DataFrame, features: list[str], category_mappings: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Converts appropriate columns to categorical type with consistent mappings.

    In training mode (category_mappings is None or empty), detects object/string columns
    with <20 unique values and converts them to categorical.
    In inference mode (category_mappings provided), applies the stored mappings.

    Args:
        df: The DataFrame to process
        features: List of feature names to consider for conversion
        category_mappings: Existing category mappings. If None or empty, training mode.
                          If populated, inference mode.

    Returns:
        Tuple of (processed DataFrame, category mappings dictionary)
    """
    if category_mappings is None:
        category_mappings = {}

    # Training mode
    if not category_mappings:
        for col in df.select_dtypes(include=["object", "string"]):
            if col in features and df[col].nunique() < 20:
                print(f"Training mode: Converting {col} to category")
                df[col] = df[col].astype("category")
                category_mappings[col] = df[col].cat.categories.tolist()

    # Inference mode
    else:
        for col, categories in category_mappings.items():
            if col in df.columns:
                print(f"Inference mode: Applying categorical mapping for {col}")
                df[col] = pd.Categorical(df[col], categories=categories)

    return df, category_mappings


def decompress_features(
    df: pd.DataFrame, features: list[str], compressed_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Decompress bitstring features into individual bit columns.

    Args:
        df: The features DataFrame
        features: Full list of feature names
        compressed_features: List of feature names to decompress (bitstrings)

    Returns:
        Tuple of (DataFrame with decompressed features, updated feature list)
    """
    # Check for any missing values in the required features
    missing_counts = df[features].isna().sum()
    if missing_counts.any():
        missing_features = missing_counts[missing_counts > 0]
        print(
            f"WARNING: Found missing values in features: {missing_features.to_dict()}. "
            "WARNING: You might want to remove/replace all NaN values before processing."
        )

    # Make a copy to avoid mutating the original list
    decompressed_features = features.copy()

    for feature in compressed_features:
        if (feature not in df.columns) or (feature not in decompressed_features):
            print(f"Feature '{feature}' not in the features list, skipping decompression.")
            continue

        # Remove the feature from the list to avoid duplication
        decompressed_features.remove(feature)

        # Handle all compressed features as bitstrings
        bit_matrix = np.array([list(bitstring) for bitstring in df[feature]], dtype=np.uint8)
        prefix = feature[:3]

        # Create all new columns at once - avoids fragmentation
        new_col_names = [f"{prefix}_{i}" for i in range(bit_matrix.shape[1])]
        new_df = pd.DataFrame(bit_matrix, columns=new_col_names, index=df.index)

        # Add to features list
        decompressed_features.extend(new_col_names)

        # Drop original column and concatenate new ones
        df = df.drop(columns=[feature])
        df = pd.concat([df, new_df], axis=1)

    return df, decompressed_features


def input_fn(input_data, content_type: str) -> pd.DataFrame:
    """Parse input data and return a DataFrame.

    Args:
        input_data: Raw input data (bytes or string)
        content_type: MIME type of the input data

    Returns:
        Parsed DataFrame

    Raises:
        ValueError: If input is empty or content_type is not supported
    """
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df: pd.DataFrame, accept_type: str) -> tuple[str, str]:
    """Convert output DataFrame to requested format.

    Args:
        output_df: DataFrame to convert
        accept_type: Requested MIME type

    Returns:
        Tuple of (formatted output string, MIME type)

    Raises:
        RuntimeError: If accept_type is not supported
    """
    if "text/csv" in accept_type:
        csv_output = output_df.fillna("N/A").to_csv(index=False)
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth target values
        y_pred: Predicted values

    Returns:
        Dictionary with keys: rmse, mae, medae, r2, spearmanr, support
    """
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearmanr": spearmanr(y_true, y_pred).correlation,
        "support": len(y_true),
    }


def print_regression_metrics(metrics: dict[str, float]) -> None:
    """Print regression metrics in the format expected by SageMaker metric definitions.

    Args:
        metrics: Dictionary of metric name -> value
    """
    print(f"rmse: {metrics['rmse']:.3f}")
    print(f"mae: {metrics['mae']:.3f}")
    print(f"medae: {metrics['medae']:.3f}")
    print(f"r2: {metrics['r2']:.3f}")
    print(f"spearmanr: {metrics['spearmanr']:.3f}")
    print(f"support: {metrics['support']}")


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], target_col: str
) -> pd.DataFrame:
    """Compute per-class classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class label names
        target_col: Name of the target column (for DataFrame output)

    Returns:
        DataFrame with columns: target_col, precision, recall, f1, support
    """
    scores = precision_recall_fscore_support(y_true, y_pred, average=None, labels=label_names)
    return pd.DataFrame(
        {
            target_col: label_names,
            "precision": scores[0],
            "recall": scores[1],
            "f1": scores[2],
            "support": scores[3],
        }
    )


def print_classification_metrics(score_df: pd.DataFrame, target_col: str, label_names: list[str]) -> None:
    """Print per-class classification metrics in the format expected by SageMaker.

    Args:
        score_df: DataFrame from compute_classification_metrics
        target_col: Name of the target column
        label_names: List of class label names
    """
    metrics = ["precision", "recall", "f1", "support"]
    for t in label_names:
        for m in metrics:
            value = score_df.loc[score_df[target_col] == t, m].iloc[0]
            print(f"Metrics:{t}:{m} {value}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> None:
    """Print confusion matrix in the format expected by SageMaker.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class label names
    """
    conf_mtx = confusion_matrix(y_true, y_pred, labels=label_names)
    for i, row_name in enumerate(label_names):
        for j, col_name in enumerate(label_names):
            value = conf_mtx[i, j]
            print(f"ConfusionMatrix:{row_name}:{col_name} {value}")
