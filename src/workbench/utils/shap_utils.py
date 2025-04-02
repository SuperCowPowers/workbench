"""Model Utilities for Workbench models"""

import logging
import numpy as np
from typing import Optional, List, Tuple

try:
    import shap
except ImportError:
    print("SHAP library is not installed. Please 'pip install shap'.")

# Workbench Imports
from workbench.utils.model_utils import xgboost_model_from_s3

# Set up the log
log = logging.getLogger("workbench")


def shap_feature_importances(workbench_model, top_n=None) -> Optional[List[Tuple[str, float]]]:
    """
    Get sorted feature importances based on SHAP values from a Workbench Model.
    Works with both regression and multi-class classification models.

    Args:
        workbench_model: Workbench Model object
        top_n: Optional integer to limit results to top N features

    Returns:
        List of tuples (feature, importance) sorted by importance (descending)
        or None if there was an error
    """

    # Get SHAP values from internal function
    features, shap_values, model, X = _calculate_shap_values(workbench_model)

    if features is None:
        return None

    # Handle different model types
    if isinstance(shap_values, list):
        # Multi-class case - average across all classes
        mean_abs_shap = np.mean([np.abs(class_shap).mean(axis=0) for class_shap in shap_values], axis=0)
    else:
        # Regression or binary classification case
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create list of (feature, importance) tuples
    shap_importance = [(features[i], float(mean_abs_shap[i])) for i in range(len(features))]

    # Sort by importance (descending)
    sorted_importance = sorted(shap_importance, key=lambda x: x[1], reverse=True)

    # Return top N if specified
    if top_n is not None and isinstance(top_n, int):
        return sorted_importance[:top_n]
    return sorted_importance


def _calculate_shap_values(workbench_model):
    """
    Internal function to calculate SHAP values for Workbench Models.
    Handles both regression and multi-class classification models.

    Args:
        workbench_model: Workbench Model object

    Returns:
        tuple containing:
        - list of feature names
        - raw shap values
        - model object
        - input data used for explanation
        or (None, None, None, None) if there was an error
    """
    from workbench.api import FeatureSet

    # Get features from workbench model
    features = workbench_model.features()

    # Get training data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.view("training").pull_dataframe()
    X = df[features]

    # Get the XGBoost model from the Workbench Model
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None, None, None, None

    # Create explainer
    explainer = shap.TreeExplainer(xgb_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    return features, shap_values, xgb_model, X


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model

    # Test the SHAP feature importances
    model = Model("abalone-regression")
    shap_values = shap_feature_importances(model)
    print("SHAP Values:")
    print(shap_values)
