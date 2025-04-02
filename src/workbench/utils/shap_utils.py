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


def shap_feature_importance(workbench_model, top_n=None) -> Optional[List[Tuple[str, float]]]:
    """
    Get feature importance data based on SHAP values from a Workbench Model.
    Works with both regression and multi-class classification models.

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
    # Get SHAP values from internal function
    features, _, model, _ = _calculate_shap_values(workbench_model)

    if features is None:
        return None

    # Ensure instance data has the right format and correct dtypes
    if hasattr(instance_data, "to_frame"):
        instance_data = instance_data.to_frame().T

    # Make sure all columns have numeric types
    for col in instance_data.columns:
        if instance_data[col].dtype == "object":
            try:
                instance_data[col] = instance_data[col].astype(float)
            except ValueError:
                log.error(f"Column {col} cannot be converted to numeric type.")
                return None

    # Create explainer and calculate SHAP values for this instance
    explainer = shap.TreeExplainer(model)
    instance_shap = explainer.shap_values(instance_data)

    # Extract feature values
    feature_values = [
        float(instance_data[features[i]].iloc[0]) if features[i] in instance_data.columns else None
        for i in range(len(features))
    ]

    # Return structured data
    return {"features": features, "shap_values": instance_shap, "feature_values": feature_values}


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
    from workbench.api import FeatureSet, Model

    # Load a model
    model = Model("abalone-regression")  # Example model

    # Example 1: Get feature importance data
    print("\n=== Feature Importance Example ===")
    importance_data = shap_feature_importance(model)
    print("SHAP feature importance:")
    for feature, importance in importance_data:
        print(f"  {feature}: {importance:.4f}")
    print("\nWhat this means: These values represent the average magnitude of each feature's")
    print("impact on model predictions. Higher values indicate more influential features.")
    print("SHAP values measure how much each feature contributes to pushing the prediction")
    print("away from the baseline (average) prediction.")

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
        print(f"Total predictions analyzed: {shap_values[0].shape[0]}")
    else:
        print("Model type: Regression or binary classification")
        print(f"SHAP matrix shape: {shap_values.shape}")
        print(f"Total predictions analyzed: {shap_values.shape[0]}")

    print("\nWhat this means: The summary data provides the raw material needed for")
    print("various SHAP visualizations. It contains the feature values from your dataset")
    print("along with their corresponding SHAP values, which represent how each feature")
    print("value impacts individual predictions. You can use this data to create summary")
    print("plots, dependence plots, or other custom visualizations.")

    # You could create visualizations here if needed
    # For example: shap.summary_plot(shap_values, summary_data['X'], feature_names=features)

    # Example 3: Get dependence data for a specific feature
    print("\n=== SHAP Dependence Data Example ===")
    top_feature = importance_data[0][0]
    dependence_data = shap_dependence_data(model, top_feature)
    feature_name = dependence_data["feature_name"]
    feature_values = dependence_data["feature_values"]
    feature_shap = dependence_data["shap_values"]

    print(f"Analyzing feature: {feature_name}")
    print(f"Number of data points: {len(feature_values)}")
    print(f"Feature value range: {min(feature_values):.3f} to {max(feature_values):.3f}")

    if isinstance(feature_shap, list):
        # Multi-class case
        print(f"SHAP value range for class 0: {min(feature_shap[0]):.3f} to {max(feature_shap[0]):.3f}")
    else:
        # Regression case
        print(f"SHAP value range: {min(feature_shap):.3f} to {max(feature_shap):.3f}")

    # Calculate correlation between feature value and SHAP value
    if isinstance(feature_shap, list):
        corr = np.corrcoef(feature_values, feature_shap[0])[0, 1]
    else:
        corr = np.corrcoef(feature_values, feature_shap)[0, 1]

    print(f"Correlation between feature value and SHAP value: {corr:.3f}")

    print("\nWhat this means: Dependence data shows how the SHAP values (model impact)")
    print(f"change as the values of {feature_name} change. A strong correlation indicates")
    print("a linear relationship, while patterns might reveal non-linear effects.")
    print("This data is useful for understanding how specific feature values affect")
    print("predictions and for detecting interactions between features.")

    # You could create visualizations here if needed

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
        print(f"  {feature}: {value}")

    # Extract and show the top contributing features
    if isinstance(shap_values, list):
        # Multi-class case (using first class for simplicity)
        contributions = [(features[i], float(shap_values[0][0][i])) for i in range(len(features))]
        print("\nTop contributions for class 0:")
    else:
        # Regression case
        contributions = [(features[i], float(shap_values[0][i])) for i in range(len(features))]
        print("\nTop feature contributions:")

    # Sort by absolute value
    sorted_contrib = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    for feature, value in sorted_contrib:
        direction = "increases" if value > 0 else "decreases"
        print(f"  {feature}: {value:.4f} ({direction} prediction)")

    print("\nWhat this means: These SHAP values explain how each feature contributes")
    print("to this specific prediction. Positive values push the prediction higher,")
    print("negative values push it lower. The magnitude shows how strongly each")
    print("feature affects this particular prediction. This helps you understand")
    print("exactly why the model made this prediction for this specific instance.")

    # You could create visualizations or formatted output here if needed
