"""XGBoost Model Utilities"""

import logging
import os
import tempfile
import tarfile
import pickle
import glob
import awswrangler as wr
from typing import Optional, List, Tuple
import hashlib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

# Workbench Imports
from workbench.utils.model_utils import load_category_mappings_from_s3
from workbench.utils.pandas_utils import convert_categorical_types

# Set up the log
log = logging.getLogger("workbench")


def xgboost_model_from_s3(model_artifact_uri: str):
    """
    Download and extract XGBoost model artifact from S3, then load the model into memory.
    Handles both direct XGBoost model files and pickled models.
    Ensures categorical feature support is enabled.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact.

    Returns:
        Loaded XGBoost model or None if unavailable.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model artifact
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

        # Extract tarball
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=tmpdir, filter="data")

        # Define model file patterns to search for (in order of preference)
        patterns = [
            # Direct XGBoost model files
            os.path.join(tmpdir, "xgboost-model"),
            os.path.join(tmpdir, "model"),
            os.path.join(tmpdir, "*.bin"),
            os.path.join(tmpdir, "**", "*model*.json"),
            os.path.join(tmpdir, "**", "rmse.json"),
            # Pickled models
            os.path.join(tmpdir, "*.pkl"),
            os.path.join(tmpdir, "**", "*.pkl"),
            os.path.join(tmpdir, "*.pickle"),
            os.path.join(tmpdir, "**", "*.pickle"),
        ]

        # Try each pattern
        for pattern in patterns:
            # Use glob to find all matching files
            for model_path in glob.glob(pattern, recursive=True):
                # Determine file type by extension
                _, ext = os.path.splitext(model_path)

                try:
                    if ext.lower() in [".pkl", ".pickle"]:
                        # Handle pickled models
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)

                        # Handle different model types
                        if isinstance(model, xgb.Booster):
                            log.important(f"Loaded XGBoost Booster from pickle: {model_path}")
                            return model
                        elif hasattr(model, "get_booster"):
                            log.important(f"Loaded XGBoost model from pipeline: {model_path}")
                            booster = model.get_booster()
                            return booster
                    else:
                        # Handle direct XGBoost model files
                        booster = xgb.Booster()
                        booster.load_model(model_path)
                        log.important(f"Loaded XGBoost model directly: {model_path}")
                        return booster
                except Exception as e:
                    log.info(f"Failed to load model from {model_path}: {e}")
                    continue  # Try the next file

    # If no model found
    log.error("No XGBoost model found in the artifact.")
    return None


def feature_importance(workbench_model, importance_type: str = "weight") -> Optional[List[Tuple[str, float]]]:
    """
    Get sorted feature importances from a Workbench Model object.

    Args:
        workbench_model: Workbench model object
        importance_type: Type of feature importance.
            Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'

    Returns:
        List of tuples (feature, importance) sorted by importance value (descending).
        Includes all features from the model, with zero importance for unused features.
        Returns None if there was an error loading the model.

    Note:
        XGBoost's get_score() only returns features with non-zero importance.
        This function ensures all model features are included in the output.
    """
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None

    # Get feature importances (only non-zero features)
    importances = xgb_model.get_score(importance_type=importance_type)

    # Get all feature names from the model
    all_features = xgb_model.feature_names

    # Create complete importance dict with zeros for missing features
    complete_importances = {feat: importances.get(feat, 0.0) for feat in all_features}

    # Convert to sorted list of tuples
    sorted_importances = sorted(complete_importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances


def _leaf_index_hash(indices):
    # Internal: Convert leaf index array to string and hash it
    leaf_str = "-".join(map(str, indices))
    hash_obj = hashlib.md5(leaf_str.encode())
    return hash_obj.hexdigest()[:10]


def add_leaf_hash(workbench_model: Any, inference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'leaf_hash' column to the dataframe representing the unique path
    through all trees in the XGBoost model.

    Args:
        workbench_model: SageMaker Workbench model object
        inference_df: DataFrame with features to run through the model

    Returns:
        DataFrame with added 'leaf_hash' column
    """
    # Extract the model
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        raise ValueError("No XGBoost model found in the artifact.")

    # Load category mappings if available
    category_mappings = load_category_mappings_from_s3(model_artifact_uri)

    # Get the features from the model and set up our XGBoost DMatrix
    features = workbench_model.features()
    X = inference_df[features]

    # Apply categorical conversions if mappings exist
    if category_mappings:
        log.info("Category mappings found. Applying categorical conversions.")
        X = convert_categorical_types(X, category_mappings)

    # Get the internal booster
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model

    # Create DMatrix with categorical features always enabled
    dmatrix = xgb.DMatrix(X, enable_categorical=True)

    # Get leaf indices for each sample across all trees
    leaf_indices = booster.predict(dmatrix, pred_leaf=True)
    leaf_hashes = [_leaf_index_hash(row) for row in leaf_indices]

    # Add the leaf hashes to the dataframe
    result_df = inference_df.copy()
    result_df["leaf_hash"] = leaf_hashes

    return result_df


def leaf_stats(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Add leaf statistics to the dataframe based on leaf_hash grouping.

    Args:
        df: DataFrame with 'leaf_hash' column and target column
        target_col: Name of the target column to compute statistics on

    Returns:
        Original DataFrame with added leaf statistic columns
    """
    if "leaf_hash" not in df.columns:
        raise ValueError("DataFrame must contain 'leaf_hash' column")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Group by leaf_hash and compute statistics with shorter syntax
    stats = (
        df.groupby("leaf_hash")[target_col]
        .agg(leaf_size="count", leaf_min="min", leaf_max="max", leaf_mean="mean", leaf_stddev="std")
        .reset_index()
    )

    # Replace NaN values in stddev with 0 (occurs when leaf_size=1)
    stats["leaf_stddev"] = stats["leaf_stddev"].fillna(0)

    # Merge statistics back to original dataframe
    result_df = df.merge(stats, on="leaf_hash", how="left")

    return result_df


def cross_fold_inference(workbench_model: Any, nfolds: int = 5) -> Dict[str, Any]:
    """
    Performs K-fold cross-validation with detailed metrics.
    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)
    Returns:
        Dictionary containing:
            - folds: Dictionary of formatted strings for each fold
            - summary_metrics: Summary metrics across folds
            - overall_metrics: Overall metrics for all folds
    """
    from workbench.api import FeatureSet

    # Load model
    model_type = workbench_model.model_type.value
    model_artifact_uri = workbench_model.model_data_url()
    loaded_booster = xgboost_model_from_s3(model_artifact_uri)
    if loaded_booster is None:
        log.error("No XGBoost model found in the artifact.")
        return {}
    # Create the model wrapper
    is_classifier = model_type == "classifier"
    xgb_model = (
        xgb.XGBClassifier(enable_categorical=True) if is_classifier else xgb.XGBRegressor(enable_categorical=True)
    )
    xgb_model._Booster = loaded_booster
    # Prepare data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.pull_dataframe()
    feature_cols = workbench_model.features()
    # Convert string features to categorical
    for col in feature_cols:
        if df[col].dtype in ["object", "string"]:
            df[col] = df[col].astype("category")
    # Split X and y
    X = df[workbench_model.features()]
    y = df[workbench_model.target()]

    # Encode target if it's a classification problem
    label_encoder = LabelEncoder() if is_classifier else None
    if label_encoder:
        y = pd.Series(label_encoder.fit_transform(y), name=workbench_model.target())
    # Prepare KFold
    kfold = (
        StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        if is_classifier
        else KFold(n_splits=nfolds, shuffle=True, random_state=42)
    )

    fold_results = []
    all_predictions = []
    all_actuals = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the model
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_val)
        all_predictions.extend(preds)
        all_actuals.extend(y_val)

        # Calculate metrics for this fold
        fold_metrics = {"fold": fold_idx + 1}

        if is_classifier:
            y_val_original = label_encoder.inverse_transform(y_val)
            preds_original = label_encoder.inverse_transform(preds.astype(int))
            scores = precision_recall_fscore_support(
                y_val_original, preds_original, average="weighted", zero_division=0
            )
            fold_metrics.update({"precision": float(scores[0]), "recall": float(scores[1]), "fscore": float(scores[2])})
        else:
            fold_metrics.update(
                {
                    "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
                    "mae": float(mean_absolute_error(y_val, preds)),
                    "r2": float(r2_score(y_val, preds)),
                }
            )

        fold_results.append(fold_metrics)
    # Calculate overall metrics
    overall_metrics = {}
    if is_classifier:
        all_actuals_original = label_encoder.inverse_transform(all_actuals)
        all_predictions_original = label_encoder.inverse_transform(all_predictions)
        scores = precision_recall_fscore_support(
            all_actuals_original, all_predictions_original, average="weighted", zero_division=0
        )
        overall_metrics.update(
            {
                "precision": float(scores[0]),
                "recall": float(scores[1]),
                "fscore": float(scores[2]),
                "confusion_matrix": confusion_matrix(
                    all_actuals_original, all_predictions_original, labels=label_encoder.classes_
                ).tolist(),
                "label_names": list(label_encoder.classes_),
            }
        )
    else:
        overall_metrics.update(
            {
                "rmse": float(np.sqrt(mean_squared_error(all_actuals, all_predictions))),
                "mae": float(mean_absolute_error(all_actuals, all_predictions)),
                "r2": float(r2_score(all_actuals, all_predictions)),
            }
        )
    # Calculate summary metrics across folds
    summary_metrics = {}
    metrics_to_aggregate = ["precision", "recall", "fscore"] if is_classifier else ["rmse", "mae", "r2"]

    for metric in metrics_to_aggregate:
        values = [fold[metric] for fold in fold_results]
        summary_metrics[metric] = f"{float(np.mean(values)):.3f} ±{float(np.std(values)):.3f}"
    # Format fold results as strings (TBD section)
    formatted_folds = {}
    for fold_data in fold_results:
        fold_key = f"Fold {fold_data['fold']}"
        if is_classifier:
            formatted_folds[fold_key] = (
                f"precision: {fold_data['precision']:.3f}  "
                f"recall: {fold_data['recall']:.3f}  "
                f"fscore: {fold_data['fscore']:.3f}"
            )
        else:
            formatted_folds[fold_key] = (
                f"rmse: {fold_data['rmse']:.3f}  mae: {fold_data['mae']:.3f}  r2: {fold_data['r2']:.3f}"
            )
    # Return the results
    return {
        "summary_metrics": summary_metrics,
        # "overall_metrics": overall_metrics,
        "folds": formatted_folds,
    }


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model, FeatureSet
    from pprint import pprint

    # Test the XGBoost model loading and feature importance
    model = Model("abalone-regression")
    features = feature_importance(model)
    print("Feature Importance:")
    print(features)

    # Test the XGBoost model loading from S3
    model_artifact_uri = model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)

    # Test with UQ Model
    uq_model = Model("aqsol-uq")
    _xgb_model = xgboost_model_from_s3(uq_model.model_data_url())

    # Test XGBoost add_leaf_hash
    input_df = FeatureSet(model.get_input()).pull_dataframe()
    leaf_df = add_leaf_hash(model, input_df)
    print("DataFrame with Leaf Hash:")
    print(leaf_df)

    # Okay, we're going to copy row 3 and insert it into row 7 to make sure the leaf_hash is the same
    input_df.iloc[7] = input_df.iloc[3]
    print("DataFrame with Leaf Hash (3 and 7 should match):")
    leaf_df = add_leaf_hash(model, input_df)
    print(leaf_df)

    # Test leaf_stats
    target_col = "class_number_of_rings"
    stats_df = leaf_stats(leaf_df, target_col)
    print("DataFrame with Leaf Statistics:")
    print(stats_df)

    print("\n=== CROSS FOLD REGRESSION EXAMPLE ===")
    model = Model("abalone-regression")
    results = cross_fold_inference(model)
    pprint(results)

    print("\n=== CROSS FOLD CLASSIFICATION EXAMPLE ===")
    model = Model("wine-classification")
    results = cross_fold_inference(model)
    pprint(results)
