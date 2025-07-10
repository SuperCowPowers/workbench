"""XGBoost Model Utilities"""

import logging
import os
import tempfile
import tarfile
import pickle
import glob
import pandas as pd
import awswrangler as wr
from typing import Optional, List, Tuple, Any
import hashlib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Union
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

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


def cross_fold_inference(workbench_model: Any, nfolds: int=5) -> Dict[str, Any]:
    """
    Performs K-fold cross-validation with detailed metrics.
    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)
    Returns:
        Dictionary containing:
            - fold_results: List of metrics for each fold
            - aggregated_metrics: Aggregated metrics across folds
            - overall_metrics: Overall metrics for all folds
            - model_type: Type of model ('classification' or 'regression')
            - nfolds: Number of folds used
    """
    from workbench.api import FeatureSet
    # Grab the XGBoost model
    model_type = workbench_model.model_type.value
    model_artifact_uri = workbench_model.model_data_url()
    loaded_booster = xgboost_model_from_s3(model_artifact_uri)  # Keep the loaded booster
    if loaded_booster is None:
        log.error("No XGBoost model found in the artifact.")
        return {}
    # Create sklearn wrapper for the loaded booster
    if model_type == 'classification':
        xgb_model = xgb.XGBClassifier(enable_categorical=True)
    else:
        xgb_model = xgb.XGBRegressor(enable_categorical=True)
    xgb_model._Booster = loaded_booster  # Assign the loaded booster to the wrapper
    # Determine model type
    model_type = workbench_model.model_type.value
    class_labels = workbench_model.get_class_labels() if model_type == "classification" else None
    # Grab all the training data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.pull_dataframe()
    # Convert string columns to categorical
    for col in df.select_dtypes(include=["object", "string"]):  # String columns
        df[col] = df[col].astype('category')
    # Split features and target
    X = df[workbench_model.features()]
    y = df[workbench_model.target()]
    # Use StratifiedKFold for classification, KFold for regression
    if model_type == 'classification':
        kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = []
    all_actuals = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Train and predict on this fold
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_val)
        # Store for overall calculations
        all_predictions.extend(preds)
        all_actuals.extend(y_val.values)
        # Calculate fold metrics
        fold_metrics = {'fold': fold_idx + 1}
        if model_type == 'classification':
            # XGBoost sklearn wrapper handles categorical encoding/decoding automatically
            scores = precision_recall_fscore_support(y_val, preds, average='weighted', zero_division=0)
            fold_metrics.update({
                'precision': float(scores[0]),
                'recall': float(scores[1]),
                'fscore': float(scores[2])
            })
        else:
            fold_metrics.update({
                'rmse': float(np.sqrt(mean_squared_error(y_val, preds))),
                'mae': float(mean_absolute_error(y_val, preds)),
                'r2': float(r2_score(y_val, preds))
            })
        fold_results.append(fold_metrics)
    # Calculate overall metrics
    overall_metrics = {}
    if model_type == 'classification':
        # XGBoost sklearn wrapper handles categorical encoding/decoding automatically
        scores = precision_recall_fscore_support(all_actuals, all_predictions, average='weighted', zero_division=0)
        overall_metrics.update({
            'precision': float(scores[0]),
            'recall': float(scores[1]),
            'fscore': float(scores[2])
        })
        # Confusion matrix - get unique labels from the data
        label_names = np.unique(np.concatenate([all_actuals, all_predictions]))
        conf_mtx = confusion_matrix(all_actuals, all_predictions, labels=label_names)
        overall_metrics['confusion_matrix'] = conf_mtx.tolist()  # Convert to list for JSON serialization
        overall_metrics['label_names'] = list(label_names)
    else:
        overall_metrics.update({
            'rmse': float(np.sqrt(mean_squared_error(all_actuals, all_predictions))),
            'mae': float(mean_absolute_error(all_actuals, all_predictions)),
            'r2': float(r2_score(all_actuals, all_predictions))
        })
    # Aggregate metrics across folds
    metrics_to_aggregate = ['precision', 'recall', 'fscore'] if model_type == 'classification' else ['rmse', 'mae', 'r2']
    aggregated_metrics = {}
    for metric in metrics_to_aggregate:
        values = [fold[metric] for fold in fold_results]
        aggregated_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    return {
        'fold_results': fold_results,
        'aggregated_metrics': aggregated_metrics,
        'overall_metrics': overall_metrics,
        'model_type': model_type,
        'nfolds': nfolds
    }


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model, FeatureSet
    from pprint import pprint

    """

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

    print("\n=== CLASSIFICATION EXAMPLE ===")
    """
    """
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y))

    # Create XGBoost classifier with custom parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=True
    )
    results = cross_fold_inference(
        model=model,
        X_train=X_df,
        y_train=y_encoded,
        nfold=5,
        label_encoder=label_encoder
    )
    pprint(results)
    print(f"Precision: {results['overall_metrics']['precision']:.3f}")
    print(f"Recall: {results['overall_metrics']['recall']:.3f}")
    print(f"F-score: {results['overall_metrics']['fscore']:.3f}")

    # Print confusion matrix
    if 'confusion_matrix' in results['overall_metrics']:
        conf_mtx = results['overall_metrics']['confusion_matrix']
        label_names = results['overall_metrics']['label_names']
        for i, row_name in enumerate(label_names):
            for j, col_name in enumerate(label_names):
                print(f"ConfusionMatrix:{row_name}:{col_name} {conf_mtx[i, j]}")
    """
    print("\n=== REGRESSION EXAMPLE ===")
    model = Model("abalone-regression")
    results = cross_fold_inference(model)
    pprint(results)

    print("\n=== CLASSIFICATION EXAMPLE ===")
    model = Model("wine-classification")
    results= cross_fold_inference(model)
    pprint(results)
