"""Model Utilities for Workbench models with explanation capabilities"""

import logging
import numpy as np
from typing import Optional, List, Tuple

try:
    import shap
except ImportError:
    print("SHAP library is not installed. Please 'pip install shap'.")

# Workbench Imports
from workbench.utils.model_utils import xgboost_model_from_s3, load_category_mappings_from_s3
from workbench.utils.pandas_utils import convert_categorical_types

# Set up the log
log = logging.getLogger("workbench")


def shap_feature_importance(workbench_model, top_n=None) -> Optional[List[Tuple[str, float]]]:
    """
    Get feature importance data based on SHAP values from a Workbench Model.
    Works with both regression and multi-class classification models.
    Currently implemented for XGBoost models.

    Args:
        workbench_model: Workbench Model object
        top_n: Optional integer to limit results to top N features

    Returns:
        List of tuples (feature, importance) sorted by importance (descending)
        or None if there was an error
    """
    # Get SHAP values from internal function
    log.important("Calculating SHAP values...")
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

    # Log the top 10 features
    log.info("Top 10 SHAP feature importances:")
    for feature, importance in sorted_importance[:10]:
        log.info(f"  {feature}: {importance:.4f}")

    # Return top N if specified
    if top_n is not None and isinstance(top_n, int):
        return sorted_importance[:top_n]
    return sorted_importance


def shap_summary_data(workbench_model) -> Optional[dict]:
    """
    Get data for SHAP summary visualization for a Workbench Model.

    Args:
        workbench_model: Workbench Model object

    Returns:
        Dictionary containing:
        - 'features': List of feature names
        - 'shap_values': SHAP values array or list of arrays for multi-class
        - 'X': DataFrame of input features
        or None if there was an error
    """
    # Get SHAP values from internal function
    features, shap_values, model, X = _calculate_shap_values(workbench_model)

    if features is None:
        return None

    # Return structured data
    return {"features": features, "shap_values": shap_values, "X": X}


def shap_dependence_data(workbench_model, feature_idx) -> Optional[dict]:
    """
    Get data for SHAP dependence plot for a specific feature.

    Args:
        workbench_model: Workbench Model object
        feature_idx: Index or name of the feature to analyze

    Returns:
        Dictionary containing:
        - 'feature_name': Name of the feature
        - 'feature_values': Array of feature values
        - 'shap_values': Array of SHAP values for the feature
        - 'features': List of all feature names for interaction selection
        - 'X': DataFrame of all features for potential interaction
        or None if there was an error
    """
    # Get SHAP values from internal function
    features, shap_values, model, X = _calculate_shap_values(workbench_model)

    if features is None:
        return None

    # Resolve feature index if name was provided
    if isinstance(feature_idx, str) and feature_idx in features:
        feature_name = feature_idx
        feature_idx = features.index(feature_idx)
    else:
        feature_name = features[feature_idx]

    # Extract feature values
    feature_values = X.iloc[:, feature_idx].values

    # Detect if feature is categorical
    is_categorical = X.iloc[:, feature_idx].dtype.name == 'category' or X.iloc[:, feature_idx].dtype == 'object'

    # Extract SHAP values for this feature
    if isinstance(shap_values, list):
        # For multi-class, return a list of arrays (one per class)
        feature_shap_values = [class_shap[:, feature_idx] for class_shap in shap_values]
    else:
        # For regression/binary, return a single array
        feature_shap_values = shap_values[:, feature_idx]

    # Return structured data
    return {
        "feature_name": feature_name,
        "feature_values": feature_values,
        "shap_values": feature_shap_values,
        "features": features,
        "X": X,
        "is_categorical": is_categorical
    }


def instance_explanation_data(workbench_model, instance_data) -> Optional[dict]:
    """
    Get SHAP explanation data for a single prediction.

    Args:
        workbench_model: Workbench Model object
        instance_data: DataFrame or Series with features for a single instance

    Returns:
        Dictionary containing:
        - 'features': List of feature names
        - 'shap_values': SHAP values for this instance (single array or list of arrays for multi-class)
        - 'feature_values': Values of features for this instance
        or None if there was an error
    """
    import xgboost as xgb

    # Get SHAP values from internal function
    features, _, model, _ = _calculate_shap_values(workbench_model)

    if features is None:
        return None

    # Ensure instance data has the right format
    if hasattr(instance_data, "to_frame"):
        instance_data = instance_data.to_frame().T

    # Load category mappings if available
    model_artifact_uri = workbench_model.model_data_url()
    category_mappings = load_category_mappings_from_s3(model_artifact_uri)

    # Apply categorical conversions if mappings exist
    if category_mappings:
        log.info("Category mappings found. Applying categorical conversions.")
        instance_data = convert_categorical_types(instance_data, category_mappings)

    # Create a DMatrix with categorical support for this instance
    dmatrix = xgb.DMatrix(instance_data, enable_categorical=True)

    # Calculate SHAP values using XGBoost's native method
    instance_shap = model.predict(dmatrix, pred_contribs=True)

    # Remove the bias term (last column) if present
    if instance_shap.shape[1] > len(features):
        instance_shap = instance_shap[0, :-1]
    else:
        instance_shap = instance_shap[0]

    # Extract feature values
    feature_values = []
    for feature in features:
        if feature in instance_data.columns:
            feature_values.append(instance_data[feature].iloc[0])
        else:
            feature_values.append(None)

    # Return structured data
    return {
        "features": features,
        "shap_values": instance_shap,
        "feature_values": feature_values
    }


def _calculate_shap_values(workbench_model):
    """
    Internal function to calculate SHAP values for Workbench Models.
    Handles both regression and multi-class classification models.
    Supports categorical features if mappings are available.

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
    import xgboost as xgb
    from workbench.api import FeatureSet

    # Get features from workbench model
    features = workbench_model.features()

    # Get training data
    fs = FeatureSet(workbench_model.get_input())
    df = fs.view("training").pull_dataframe()
    X = df[features].copy()

    # Get the XGBoost model from the Workbench Model
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None, None, None, None

    # Load category mappings if available
    category_mappings = load_category_mappings_from_s3(model_artifact_uri)

    # Apply categorical conversions if mappings exist
    if category_mappings:
        log.info("Category mappings found. Applying categorical conversions.")
        X = convert_categorical_types(X, category_mappings)

    # Create a DMatrix with categorical support
    dmatrix = xgb.DMatrix(X, enable_categorical=True)

    # Use XGBoost's built-in SHAP calculation
    shap_values = xgb_model.predict(dmatrix, pred_contribs=True)

    # Remove the bias term (last column) if present
    if shap_values.shape[1] > len(features):
        shap_values = shap_values[:, :-1]

    return features, shap_values, xgb_model, X


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import FeatureSet, Model

    # Load a model
    model = Model("test-regression")  # Example model

    # Example 1: Get feature importance data
    print("\n=== Feature Importance Example ===")
    importance_data = shap_feature_importance(model)
    print("SHAP feature importance:")
    for feature, importance in importance_data:
        print(f"  {feature}: {importance:.4f}")
    print("\nWhat this means: These values represent the average magnitude of each feature's")
    print("impact on model predictions. Higher values indicate more influential features.")

    # Example 2: Get summary data
    print("\n=== SHAP Summary Data Example ===")
    summary_data = shap_summary_data(model)
    features = summary_data["features"]
    shap_values = summary_data["shap_values"]
    X = summary_data["X"]

    print("SHAP features summary:")
    print(f"Number of features analyzed: {len(features)}")
    print(f"Feature names: {', '.join(features)}")

    if isinstance(shap_values, list):
        print(f"Model type: Classification with {len(shap_values)} classes")
        print(f"SHAP matrix shape for class 0: {shap_values[0].shape}")
    else:
        print("Model type: Regression or binary classification")
        print(f"SHAP matrix shape: {shap_values.shape}")

    # Example 3: Get dependence data for a specific feature
    print("\n=== SHAP Dependence Data Example ===")
    top_feature = importance_data[0][0]
    dependence_data = shap_dependence_data(model, top_feature)
    feature_name = dependence_data["feature_name"]
    feature_values = dependence_data["feature_values"]
    feature_shap = dependence_data["shap_values"]
    is_categorical = dependence_data.get("is_categorical", False)

    print(f"Analyzing feature: {feature_name}")
    print(f"Number of data points: {len(feature_values)}")

    # Handle display based on feature type
    if is_categorical:
        try:
            # Convert values to strings first to avoid comparison issues
            str_values = [str(v) for v in feature_values]
            unique_values = set(str_values)
            print(f"Feature is categorical with {len(unique_values)} unique values")
        except:
            print("Feature is categorical")
    else:
        print(f"Feature value range: {min(feature_values):.3f} to {max(feature_values):.3f}")

        # Only calculate correlation for numeric features
        if isinstance(feature_shap, list):
            corr = np.corrcoef(feature_values, feature_shap[0])[0, 1]
        else:
            corr = np.corrcoef(feature_values, feature_shap)[0, 1]
        print(f"Correlation between feature value and SHAP value: {corr:.3f}")

    # Example 4: Get instance explanation data
    print("\n=== Instance Explanation Data Example ===")
    fs = FeatureSet(model.get_input())
    sample_data = fs.view("training").pull_dataframe()[features][0:1]
    explanation_data = instance_explanation_data(model, sample_data)
    print("Explaining prediction for a single instance:")
    print(f"Number of features: {len(explanation_data['features'])}")

    features = explanation_data["features"]
    feature_values = explanation_data["feature_values"]
    shap_values = explanation_data["shap_values"]

    # Display the instance's feature values
    print("\nFeature values for this instance:")
    for i, (feature, value) in enumerate(zip(features, feature_values)):
        # Simple formatting based on type
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            formatted_value = f"{value:.4f}" if value != int(value) else str(int(value))
        else:
            formatted_value = str(value)
        print(f"  {feature}: {formatted_value}")

    # Extract and show the top contributing features
    if isinstance(shap_values, list):
        # Multi-class case (using first class for simplicity)
        contributions = [(features[i], float(shap_values[0][i])) for i in range(len(features))]
        print("\nTop contributions for class 0:")
    else:
        # Regression case
        contributions = [(features[i], float(shap_values[i])) for i in range(len(features))]
        print("\nTop feature contributions:")

    # Sort by absolute value
    sorted_contrib = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    for feature, value in sorted_contrib:
        direction = "increases" if value > 0 else "decreases"
        print(f"  {feature}: {value:.4f} ({direction} prediction)")