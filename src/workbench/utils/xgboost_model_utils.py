"""XGBoost Model Utilities"""

import logging
import os
import tempfile
import tarfile
import joblib
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
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=tmpdir, filter="data")

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


def cross_fold_inference(workbench_model: Any, nfolds: int = 5) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs K-fold cross-validation with detailed metrics.
    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)
    Returns:
        Tuple of:
            - Dictionary containing:
                - folds: Dictionary of formatted strings for each fold
                - summary_metrics: Summary metrics across folds
            - DataFrame with columns: id, target, prediction (out-of-fold predictions for all samples)
    """
    from workbench.api import FeatureSet

    # Load model
    model_artifact_uri = workbench_model.model_data_url()
    loaded_model = xgboost_model_from_s3(model_artifact_uri)
    if loaded_model is None:
        log.error("No XGBoost model found in the artifact.")
        return {}, pd.DataFrame()

    # Check if we got a full sklearn model or need to create one
    if isinstance(loaded_model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        xgb_model = loaded_model
        is_classifier = isinstance(xgb_model, xgb.XGBClassifier)
    elif isinstance(loaded_model, xgb.Booster):
        # Legacy: got a booster, need to wrap it
        log.warning("Deprecated: Loaded model is a Booster, wrapping in sklearn model.")
        is_classifier = workbench_model.model_type.value == "classifier"
        xgb_model = (
            xgb.XGBClassifier(enable_categorical=True) if is_classifier else xgb.XGBRegressor(enable_categorical=True)
        )
        xgb_model._Booster = loaded_model
    else:
        log.error(f"Unexpected model type: {type(loaded_model)}")
        return {}, pd.DataFrame()

    # Prepare data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.view("training").pull_dataframe()

    # Get id column - assuming FeatureSet has an id_column attribute or similar
    id_col = fs.id_column
    target_col = workbench_model.target()
    feature_cols = workbench_model.features()

    # Convert string features to categorical
    for col in feature_cols:
        if df[col].dtype in ["object", "string"]:
            df[col] = df[col].astype("category")

    X = df[feature_cols]
    y = df[target_col]
    ids = df[id_col]

    # Encode target if classifier
    label_encoder = LabelEncoder() if is_classifier else None
    if label_encoder:
        y_encoded = label_encoder.fit_transform(y)
        y_for_cv = pd.Series(y_encoded, index=y.index, name=target_col)
    else:
        y_for_cv = y

    # Prepare KFold
    kfold = (StratifiedKFold if is_classifier else KFold)(n_splits=nfolds, shuffle=True, random_state=42)

    # Initialize results collection
    fold_metrics = []
    predictions_df = pd.DataFrame({id_col: ids, target_col: y})  # Keep original values
    # Note: 'prediction' column will be created automatically with correct dtype

    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y_for_cv), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_for_cv.iloc[train_idx], y_for_cv.iloc[val_idx]

        # Train and predict
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_val)

        # Store predictions (decode if classifier)
        val_indices = X_val.index
        if is_classifier:
            predictions_df.loc[val_indices, "prediction"] = label_encoder.inverse_transform(preds.astype(int))
        else:
            predictions_df.loc[val_indices, "prediction"] = preds

        # Calculate fold metrics
        if is_classifier:
            y_val_orig = label_encoder.inverse_transform(y_val)
            preds_orig = label_encoder.inverse_transform(preds.astype(int))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val_orig, preds_orig, average="weighted", zero_division=0
            )
            fold_metrics.append({"fold": fold_idx, "precision": prec, "recall": rec, "fscore": f1})
        else:
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "rmse": np.sqrt(mean_squared_error(y_val, preds)),
                    "mae": mean_absolute_error(y_val, preds),
                    "r2": r2_score(y_val, preds),
                }
            )

    # Calculate summary metrics (mean ± std)
    fold_df = pd.DataFrame(fold_metrics)
    metric_names = ["precision", "recall", "fscore"] if is_classifier else ["rmse", "mae", "r2"]
    summary_metrics = {metric: f"{fold_df[metric].mean():.3f} ±{fold_df[metric].std():.3f}" for metric in metric_names}

    # Format fold results for display
    formatted_folds = {}
    for _, row in fold_df.iterrows():
        fold_key = f"Fold {int(row['fold'])}"
        if is_classifier:
            formatted_folds[fold_key] = (
                f"precision: {row['precision']:.3f}  " f"recall: {row['recall']:.3f}  " f"fscore: {row['fscore']:.3f}"
            )
        else:
            formatted_folds[fold_key] = f"rmse: {row['rmse']:.3f}  " f"mae: {row['mae']:.3f}  " f"r2: {row['r2']:.3f}"

    # Build return dictionary
    metrics_dict = {"summary_metrics": summary_metrics, "folds": formatted_folds}

    return metrics_dict, predictions_df


def leave_one_out_inference(workbench_model: Any) -> pd.DataFrame:
    """
    Performs leave-one-out cross-validation (parallelized).
    For datasets > 1000 rows, first identifies top 100 worst predictions via 10-fold CV,
    then performs true leave-one-out on those 100 samples.
    Each model trains on ALL data except one sample.
    """
    from workbench.api import FeatureSet
    from joblib import Parallel, delayed
    from tqdm import tqdm

    def train_and_predict_one(model_params, is_classifier, X, y, train_idx, val_idx):
        """Train on train_idx, predict on val_idx."""
        model = xgb.XGBClassifier(**model_params) if is_classifier else xgb.XGBRegressor(**model_params)
        model.fit(X[train_idx], y[train_idx])
        return model.predict(X[val_idx])[0]

    # Load model and get params
    model_artifact_uri = workbench_model.model_data_url()
    loaded_model = xgboost_model_from_s3(model_artifact_uri)
    if loaded_model is None:
        log.error("No XGBoost model found in the artifact.")
        return pd.DataFrame()

    if isinstance(loaded_model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        is_classifier = isinstance(loaded_model, xgb.XGBClassifier)
        model_params = loaded_model.get_params()
    elif isinstance(loaded_model, xgb.Booster):
        log.warning("Deprecated: Loaded model is a Booster, wrapping in sklearn model.")
        is_classifier = workbench_model.model_type.value == "classifier"
        model_params = {"enable_categorical": True}
    else:
        log.error(f"Unexpected model type: {type(loaded_model)}")
        return pd.DataFrame()

    # Load and prepare data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.view("training").pull_dataframe()
    id_col = fs.id_column
    target_col = workbench_model.target()
    feature_cols = workbench_model.features()

    # Convert string features to categorical
    for col in feature_cols:
        if df[col].dtype in ["object", "string"]:
            df[col] = df[col].astype("category")

    # Determine which samples to run LOO on
    if len(df) > 1000:
        log.important(f"Dataset has {len(df)} rows. Running 10-fold CV to identify top 1000 worst predictions...")
        _, predictions_df = cross_fold_inference(workbench_model, nfolds=10)
        predictions_df["residual_abs"] = np.abs(predictions_df[target_col] - predictions_df["prediction"])
        worst_samples = predictions_df.nlargest(1000, "residual_abs")
        worst_ids = worst_samples[id_col].values
        loo_indices = df[df[id_col].isin(worst_ids)].index.values
        log.important(f"Running leave-one-out CV on 1000 worst samples. Each model trains on {len(df)-1} rows...")
    else:
        log.important(f"Running leave-one-out CV on all {len(df)} samples...")
        loo_indices = df.index.values

    # Prepare full dataset for training
    X_full = df[feature_cols].values
    y_full = df[target_col].values

    # Encode target if classifier
    label_encoder = LabelEncoder() if is_classifier else None
    if label_encoder:
        y_full = label_encoder.fit_transform(y_full)

    # Generate LOO splits
    splits = []
    for loo_idx in loo_indices:
        train_idx = np.delete(np.arange(len(X_full)), loo_idx)
        val_idx = np.array([loo_idx])
        splits.append((train_idx, val_idx))

    # Parallel execution
    predictions = Parallel(n_jobs=4)(
        delayed(train_and_predict_one)(model_params, is_classifier, X_full, y_full, train_idx, val_idx)
        for train_idx, val_idx in tqdm(splits, desc="LOO CV")
    )

    # Build results dataframe
    predictions_array = np.array(predictions)
    if label_encoder:
        predictions_array = label_encoder.inverse_transform(predictions_array.astype(int))

    predictions_df = pd.DataFrame(
        {
            id_col: df.loc[loo_indices, id_col].values,
            target_col: df.loc[loo_indices, target_col].values,
            "prediction": predictions_array,
        }
    )

    predictions_df["residual_abs"] = np.abs(predictions_df[target_col] - predictions_df["prediction"])

    return predictions_df


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

    # Verify enable_categorical is preserved (for debugging/confidence)
    print(f"Model parameters: {xgb_model.get_params()}")
    print(f"enable_categorical: {xgb_model.enable_categorical}")

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
    results, df = cross_fold_inference(model)
    pprint(results)
    print(df.head())

    print("\n=== CROSS FOLD CLASSIFICATION EXAMPLE ===")
    model = Model("wine-classification")
    results, df = cross_fold_inference(model)
    pprint(results)
    print(df.head())
