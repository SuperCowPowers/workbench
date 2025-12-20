"""XGBoost Model Utilities"""

import glob
import hashlib
import logging
import os
import pickle
import tempfile
from typing import Any, List, Optional, Tuple

import awswrangler as wr
import joblib
import pandas as pd
import xgboost as xgb

# Workbench Imports
from workbench.utils.aws_utils import pull_s3_data
from workbench.utils.metrics_utils import compute_metrics_from_predictions
from workbench.utils.model_utils import load_category_mappings_from_s3, safe_extract_tarfile
from workbench.utils.pandas_utils import convert_categorical_types

# Set up the log
log = logging.getLogger("workbench")


def xgboost_model_from_s3(model_artifact_uri: str):
    """
    Download and extract XGBoost model artifact from S3, then load the model into memory.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact.

    Returns:
        Loaded XGBoost model (XGBClassifier, XGBRegressor, or Booster) or None if unavailable.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model artifact
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)

        # Extract tarball
        safe_extract_tarfile(local_tar_path, tmpdir)

        # Define model file patterns to search for (in order of preference)
        patterns = [
            # Joblib models (preferred - preserves everything)
            os.path.join(tmpdir, "*model*.joblib"),
            os.path.join(tmpdir, "xgb*.joblib"),
            os.path.join(tmpdir, "**", "*model*.joblib"),
            os.path.join(tmpdir, "**", "xgb*.joblib"),
            # Pickle models (also preserves everything)
            os.path.join(tmpdir, "*model*.pkl"),
            os.path.join(tmpdir, "xgb*.pkl"),
            os.path.join(tmpdir, "**", "*model*.pkl"),
            os.path.join(tmpdir, "**", "xgb*.pkl"),
            # JSON models (fallback - requires reconstruction)
            os.path.join(tmpdir, "*model*.json"),
            os.path.join(tmpdir, "xgb*.json"),
            os.path.join(tmpdir, "**", "*model*.json"),
            os.path.join(tmpdir, "**", "xgb*.json"),
        ]

        # Try each pattern
        for pattern in patterns:
            for model_path in glob.glob(pattern, recursive=True):
                # Skip files that are clearly not XGBoost models
                filename = os.path.basename(model_path).lower()
                if any(skip in filename for skip in ["label_encoder", "scaler", "preprocessor", "transformer"]):
                    log.debug(f"Skipping non-model file: {model_path}")
                    continue

                _, ext = os.path.splitext(model_path)

                try:
                    if ext == ".joblib":
                        model = joblib.load(model_path)
                        # Verify it's actually an XGBoost model
                        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor, xgb.Booster)):
                            log.important(f"Loaded XGBoost model from joblib: {model_path}")
                            return model
                        else:
                            log.debug(f"Skipping non-XGBoost object from {model_path}: {type(model)}")

                    elif ext in [".pkl", ".pickle"]:
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        # Verify it's actually an XGBoost model
                        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor, xgb.Booster)):
                            log.important(f"Loaded XGBoost model from pickle: {model_path}")
                            return model
                        else:
                            log.debug(f"Skipping non-XGBoost object from {model_path}: {type(model)}")

                    elif ext == ".json":
                        # JSON files should be XGBoost models by definition
                        booster = xgb.Booster()
                        booster.load_model(model_path)
                        log.important(f"Loaded XGBoost booster from JSON: {model_path}")
                        return booster

                except Exception as e:
                    log.debug(f"Failed to load {model_path}: {e}")
                    continue

    log.error("No XGBoost model found in the artifact.")
    return None


def feature_importance(workbench_model, importance_type: str = "gain") -> Optional[List[Tuple[str, float]]]:
    """
    Get sorted feature importances from a Workbench Model object.

    Args:
        workbench_model: Workbench model object
        importance_type: Type of feature importance. Options:
            - 'gain' (default): Average improvement in loss/objective when feature is used.
                     Best for understanding predictive power of features.
            - 'weight': Number of times a feature appears in trees (split count).
                       Useful for understanding model complexity and feature usage frequency.
            - 'cover': Average number of samples affected when feature is used.
                      Shows the relative quantity of observations related to this feature.
            - 'total_gain': Total improvement in loss/objective across all splits.
                           Similar to 'gain' but not averaged (can be biased toward frequent features).
            - 'total_cover': Total number of samples affected across all splits.
                            Similar to 'cover' but not averaged.

    Returns:
        List of tuples (feature, importance) sorted by importance value (descending).
        Includes all features from the model, with zero importance for unused features.
        Returns None if there was an error loading the model.

    Note:
        XGBoost's get_score() only returns features with non-zero importance.
        This function ensures all model features are included in the output,
        adding zero values for features that weren't used in any tree splits.
    """
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None

    # Check if we got a full sklearn model or just a booster (for backwards compatibility)
    if hasattr(xgb_model, "get_booster"):
        # Full sklearn model - get the booster for feature importance
        booster = xgb_model.get_booster()
        all_features = booster.feature_names
    else:
        # Already a booster (legacy JSON load)
        booster = xgb_model
        all_features = xgb_model.feature_names

    # Get feature importances (only non-zero features)
    importances = booster.get_score(importance_type=importance_type)

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


def pull_cv_results(workbench_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cross-validation results from AWS training artifacts.

    This retrieves the validation predictions saved during model training and
    computes metrics directly from them. For XGBoost models trained with
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
    """Exercise the Model Utilities"""
    from workbench.api import Model

    # Test the XGBoost model loading and feature importance
    model = Model("abalone-regression")
    features = feature_importance(model)
    print("Feature Importance:")
    print(features)

    # Test the XGBoost model loading from S3
    model_artifact_uri = model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)

    # Verify enable_categorical is preserved (for debugging/confidence)
    print(f"Model parameters: {xgb_model.get_params()}")
    print(f"enable_categorical: {xgb_model.enable_categorical}")

    print("\n=== PULL CV RESULTS EXAMPLE ===")
    model = Model("abalone-regression")
    metrics_df, predictions_df = pull_cv_results(model)
    print(f"\nMetrics:\n{metrics_df}")
    print(f"\nPredictions shape: {predictions_df.shape}")
    print(f"Predictions columns: {predictions_df.columns.tolist()}")
    print(predictions_df.head())

    # Test on a Classifier model
    print("\n=== CLASSIFIER MODEL TEST ===")
    model = Model("wine-classification")
    features = feature_importance(model)
    print("Feature Importance:")
    print(features)
    metrics_df, predictions_df = pull_cv_results(model)
    print(f"\nMetrics:\n{metrics_df}")
    print(f"\nPredictions shape: {predictions_df.shape}")
    print(f"Predictions columns: {predictions_df.columns.tolist()}")
    print(predictions_df.head())
