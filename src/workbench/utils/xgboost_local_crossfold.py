"""XGBoost Local Cross-Fold Validation Utilities

This module contains functions for running cross-validation locally on XGBoost models.
For most use cases, prefer using pull_cv_results() from xgboost_model_utils.py which
retrieves the CV results that were saved during training on SageMaker.

These local cross-fold functions are useful for:
- Re-running CV with different fold counts
- Leave-one-out cross-validation
- Custom CV experiments
"""

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from workbench.utils.metrics_utils import compute_metrics_from_predictions
from workbench.utils.pandas_utils import expand_proba_column
from workbench.utils.xgboost_model_utils import xgboost_model_from_s3

log = logging.getLogger("workbench")


def cross_fold_inference(workbench_model: Any, nfolds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs K-fold cross-validation locally with detailed metrics.

    Note: For most use cases, prefer using pull_cv_results() from xgboost_model_utils.py
    which retrieves the CV results that were saved during training.

    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)
    Returns:
        Tuple of:
            - DataFrame with per-class metrics (and 'all' row for overall metrics)
            - DataFrame with columns: id, target, prediction, and *_proba columns (for classifiers)
    """
    from workbench.api import FeatureSet

    # Load model
    model_artifact_uri = workbench_model.model_data_url()
    loaded_model = xgboost_model_from_s3(model_artifact_uri)
    if loaded_model is None:
        log.error("No XGBoost model found in the artifact.")
        return pd.DataFrame(), pd.DataFrame()

    # Check if we got a full sklearn model or need to create one
    if isinstance(loaded_model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        is_classifier = isinstance(loaded_model, xgb.XGBClassifier)

        # Get the model's hyperparameters and ensure enable_categorical=True
        params = loaded_model.get_params()
        params["enable_categorical"] = True

        # Create new model with same params but enable_categorical=True
        if is_classifier:
            xgb_model = xgb.XGBClassifier(**params)
        else:
            xgb_model = xgb.XGBRegressor(**params)

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
        return pd.DataFrame(), pd.DataFrame()

    # Prepare data
    fs = FeatureSet(workbench_model.get_input())
    df = workbench_model.training_view().pull_dataframe()

    # Extract sample weights if present
    sample_weights = df.get("sample_weight")
    if sample_weights is not None:
        log.info(f"Using sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}")

    # Get columns
    id_col = fs.id_column
    target_col = workbench_model.target()
    feature_cols = workbench_model.features()
    print(f"Target column: {target_col}")
    print(f"Feature columns: {len(feature_cols)} features")

    # Convert string[python] to object, then to category for XGBoost compatibility
    for col in feature_cols:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("object").astype("category")

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

    # Initialize predictions DataFrame
    predictions_df = pd.DataFrame({id_col: ids, target_col: y})

    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y_for_cv), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y_for_cv.iloc[train_idx]

        # Get sample weights for training fold
        weights_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

        # Train and predict
        xgb_model.fit(X_train, y_train, sample_weight=weights_train)
        preds = xgb_model.predict(X_val)

        # Store predictions (decode if classifier)
        val_indices = X_val.index
        if is_classifier:
            predictions_df.loc[val_indices, "prediction"] = label_encoder.inverse_transform(preds.astype(int))
            y_proba = xgb_model.predict_proba(X_val)
            predictions_df.loc[val_indices, "pred_proba"] = pd.Series(y_proba.tolist(), index=val_indices)
        else:
            predictions_df.loc[val_indices, "prediction"] = preds

    # Expand proba columns for classifiers
    if is_classifier:
        predictions_df = expand_proba_column(predictions_df, label_encoder.classes_)

    # Compute metrics from the complete out-of-fold predictions
    class_labels = list(label_encoder.classes_) if is_classifier else None
    metrics_df = compute_metrics_from_predictions(predictions_df, target_col, class_labels)

    return metrics_df, predictions_df


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
    df = workbench_model.training_view().pull_dataframe()
    id_col = fs.id_column
    target_col = workbench_model.target()
    feature_cols = workbench_model.features()

    # Convert string[python] to object, then to category for XGBoost compatibility
    # This avoids XGBoost's issue with pandas 2.x string[python] dtype in categorical categories
    for col in feature_cols:
        if pd.api.types.is_string_dtype(df[col]):
            # Double conversion: string[python] -> object -> category
            df[col] = df[col].astype("object").astype("category")

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
    """Exercise the Local Cross-Fold Utilities"""
    from workbench.api import Model
    from pprint import pprint

    print("\n=== LOCAL CROSS FOLD REGRESSION EXAMPLE ===")
    model = Model("abalone-regression")
    results, df = cross_fold_inference(model)
    pprint(results)
    print(df.head())

    print("\n=== LOCAL CROSS FOLD CLASSIFICATION EXAMPLE ===")
    model = Model("wine-classification")
    results, df = cross_fold_inference(model)
    pprint(results)
    print(df.head())
