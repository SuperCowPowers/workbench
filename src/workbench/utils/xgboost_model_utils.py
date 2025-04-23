"""XGBoost Model Utilities"""

import logging
import os
import json
import tempfile
import tarfile
import pickle
import glob
import numpy as np
import awswrangler as wr
from typing import Optional, List, Tuple, Dict, Any
import xgboost as xgb

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
    Get sorted feature importances from an Workbench Model object.

    Args:
        workbench_model: Workbench model object
        importance_type: Type of feature importance.
            Options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'

    Returns:
        List of tuples (feature, importance) sorted by importance value (descending)
        or None if there was an error
    """
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None

    # Get feature importances
    importances = xgb_model.get_score(importance_type=importance_type)

    # Convert to sorted list of tuples (feature, importance)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances


def get_xgboost_trees(workbench_model: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract the internal tree structure from a Workbench XGBoost model.

    Args:
        workbench_model: SageMaker Workbench model object

    Returns:
        List of tree root nodes or None if model couldn't be loaded
    """
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None

    # Get the internal booster
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model

    # Dump the model as JSON
    model_json = booster.get_dump(dump_format="json")

    # Parse the JSON strings into Python dictionaries (root nodes)
    tree_roots = [json.loads(tree) for tree in model_json]

    return tree_roots


def create_leaf_map(trees: List[Dict[str, Any]]) -> List[Dict[int, Dict[str, Any]]]:
    """
    Create a mapping of leaf indices to predictions and paths for each tree.

    Args:
        trees: List of tree root nodes from XGBoost model

    Returns:
        List of dictionaries mapping leaf indices to prediction data for each tree
    """
    leaf_maps = []

    for tree in trees:
        # Get leaf mapping for this tree
        tree_leaf_map = {}

        def map_leaves(node, path=None, leaf_idx=None):
            if path is None:
                path = []
                leaf_idx = [0]  # Use a list for mutable counter

            if "leaf" in node:
                # This is a leaf node - record its prediction and index
                tree_leaf_map[leaf_idx[0]] = {
                    "path": path.copy(),
                    "prediction": node["leaf"],
                    "sample_targets": [],  # Will store target values of samples in this leaf
                }
                leaf_idx[0] += 1
            else:
                # Navigate children
                map_leaves(node["children"][0], path + [0], leaf_idx)
                map_leaves(node["children"][1], path + [1], leaf_idx)

        map_leaves(tree)
        leaf_maps.append(tree_leaf_map)

    return leaf_maps


def compute_regression_confidence(
    X_train: np.ndarray, y_train: np.ndarray, xgb_model: xgb.Booster, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence scores for regression predictions based on leaf node variance.

    Args:
        X_train: Training features
        y_train: Training target values
        xgb_model: Trained XGBoost model
        X_test: Test features to predict

    Returns:
        Tuple of (predictions, confidence_scores)
    """
    # Get tree structure
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
    model_json = booster.get_dump(dump_format="json")
    trees = [json.loads(tree) for tree in model_json]

    # Create leaf mapping
    leaf_maps = create_leaf_map(trees)

    # Find which training samples fall into which leaf nodes
    train_leaf_indices = booster.predict(xgb.DMatrix(X_train), pred_leaf=True)

    # Store target values in each leaf node
    for i, sample_leaves in enumerate(train_leaf_indices):
        for tree_idx, leaf_idx in enumerate(sample_leaves):
            leaf_maps[tree_idx][leaf_idx]["sample_targets"].append(y_train[i])

    # Calculate statistics for each leaf node
    for tree_map in leaf_maps:
        for leaf_data in tree_map.values():
            targets = leaf_data["sample_targets"]
            if targets:
                leaf_data["target_mean"] = np.mean(targets)
                leaf_data["target_std"] = np.std(targets)
                leaf_data["target_count"] = len(targets)
            else:
                # Handle empty leaves
                leaf_data["target_mean"] = 0
                leaf_data["target_std"] = float("inf")
                leaf_data["target_count"] = 0

    # Predict for test data
    test_preds = booster.predict(xgb.DMatrix(X_test))
    test_leaf_indices = booster.predict(xgb.DMatrix(X_test), pred_leaf=True)

    # Calculate confidence scores (inverse of weighted standard deviation)
    confidence_scores = np.zeros(len(X_test))
    for i, sample_leaves in enumerate(test_leaf_indices):
        leaf_stds = []
        leaf_counts = []

        for tree_idx, leaf_idx in enumerate(sample_leaves):
            leaf_data = leaf_maps[tree_idx][leaf_idx]
            leaf_stds.append(leaf_data["target_std"])
            leaf_counts.append(leaf_data["target_count"])

        # Weight by sample count and invert (higher values = more confidence)
        weighted_std = np.average(leaf_stds, weights=leaf_counts) if np.sum(leaf_counts) > 0 else float("inf")
        confidence_scores[i] = 1.0 / (weighted_std + 1e-6)  # Add small constant to avoid division by zero

    return test_preds, confidence_scores


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model

    # Test the XGBoost model loading and feature importance
    model = Model("abalone-regression")
    features = feature_importance(model)
    print("Feature Importance:")
    print(features)

    # Test XGBoost internal tree structure
    trees = get_xgboost_trees(model)

    # Test creating leaf map
    leaf_map = create_leaf_map(trees)
    print("Leaf Map:")
    print(leaf_map)

    # Test the prediction for one sample
    # df = FeatureSet("abalone-regression").pull_dataframe(limit=1)
