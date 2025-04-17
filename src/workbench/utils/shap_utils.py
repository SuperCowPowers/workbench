"""Model Utilities for Workbench models with explanation capabilities"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union

# Workbench Imports
from workbench.utils.model_utils import xgboost_model_from_s3, load_category_mappings_from_s3
from workbench.utils.pandas_utils import convert_categorical_types

# Set up the log
log = logging.getLogger("workbench")


def shap_feature_importance(workbench_model, shap_data=None, top_n=None) -> Optional[List[Tuple[str, float]]]:
    """
    Get feature importance data based on SHAP values from a Workbench Model.
    Works with both regression and multi-class classification models.

    Args:
        workbench_model: Workbench Model object
        shap_data: Optional pre-calculated SHAP values data from shap_values_data()
        top_n: Optional integer to limit results to top N features

    Returns:
        List of tuples (feature, importance) sorted by importance (descending)
        or None if there was an error
    """
    # Get SHAP data if not provided
    if shap_data is None:
        log.important("Calculating SHAP values...")
        shap_data = shap_values_data(workbench_model)
        if shap_data is None:
            return None

    # Get feature names directly from the model
    features = workbench_model.features()

    # Check if multi-class (dictionary of DataFrames) or single-class (DataFrame)
    is_multiclass = isinstance(shap_data, dict)
    if is_multiclass:
        # For multiclass, calculate mean absolute SHAP value across all classes
        mean_abs_shap = np.mean([np.abs(df[features].values).mean(axis=0) for df in shap_data.values()], axis=0)
    else:
        # For single class, get mean abs values
        mean_abs_shap = np.abs(shap_data[features].values).mean(axis=0)

    # Create list of (feature, importance) tuples
    shap_importance = [(features[i], float(mean_abs_shap[i])) for i in range(len(features))]

    # Sort by importance (descending)
    sorted_importance = sorted(shap_importance, key=lambda x: x[1], reverse=True)

    # Return top N if specified
    if top_n is not None and isinstance(top_n, int):
        return sorted_importance[:top_n]
    return sorted_importance


def shap_values_data(workbench_model) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Get SHAP explanation data for all instances in the training data.
    Handles both regression/binary classification and multi-class models.

    Args:
        workbench_model: Workbench Model object

    Returns:
        For regression/binary: DataFrame with SHAP values, one row per instance, columns are features
        For multi-class: Dictionary of DataFrames, one per class, each with SHAP values

    Note:
        The ID column is always included as the first column of each DataFrame.
    """
    # Get all data from internal function
    features, shap_values, _, ids = _calculate_shap_values(workbench_model)
    if features is None:
        return None

    # Check if we have a multi-class model (SHAP values will have 3 dimensions)
    is_multiclass = shap_values.shape[1] > 1
    if is_multiclass:
        # For multi-class models, return a dictionary of DataFrames (one per class)
        # The second dimension is the number of classes
        num_classes = shap_values.shape[1]
        class_labels = workbench_model.class_labels()
        if num_classes != len(class_labels):
            log.error("Mismatch between number of classes in SHAP values and Workbench model.")
            return None
        result = {}

        # Create a DataFrame for EACH class
        for idx, label in enumerate(class_labels):
            class_df = pd.DataFrame(shap_values[:, idx, :], columns=features)
            class_df.insert(0, ids.name, ids)
            result[label] = class_df
        return result

    # For regression or binary classification models (single class)
    else:
        # Extract the single class from the 3D array
        single_class_values = shap_values[:, 0, :]
        result_df = pd.DataFrame(single_class_values, columns=features)
        result_df.insert(0, ids.name, ids)
        return result_df


def _calculate_shap_values(workbench_model):
    """
    Internal function to calculate SHAP values for Workbench Models.
    Handles both regression and multi-class classification models.
    Supports categorical features if mappings are available.

    Args:
        workbench_model: Workbench Model object

    Returns:
        - list of feature names
        - raw shap values
        - input data used for explanation
        - ids of the input data
        or (None, None, None, None) if there was an error
    """
    import xgboost as xgb
    from workbench.api import FeatureSet

    # Get features from workbench model
    features = workbench_model.features()

    # Get input data for the model
    fs = FeatureSet(workbench_model.get_input())
    df = fs.pull_dataframe()
    X = df[features].copy()
    ids = df[fs.id_column]

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
    shap_values = xgb_model.predict(dmatrix, pred_contribs=True, strict_shape=True)

    # Return the feature names, SHAP values, input data, and IDs
    features_with_bias = features + ["bias"]
    return features_with_bias, shap_values, X, ids


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model

    # Set pandas display options
    pd.options.display.max_columns = 20
    pd.options.display.max_colwidth = 200
    pd.options.display.width = 1400

    # Test a regression model
    model = Model("test-regression")

    # Example 1: Get feature importance data
    print("\n=== Feature Importance Example ===")
    importance_data = shap_feature_importance(model)
    print("SHAP feature importance:")
    for feature, importance in importance_data:
        print(f"  {feature}: {importance:.4f}")
    print("\nWhat this means: These values represent the average magnitude of each feature's")
    print("impact on model predictions. Higher values indicate more influential features.")

    # Get instance explanation data
    print("\n=== Instance Explanation Data Example (Regression) ===")
    shap_df = shap_values_data(model)
    print(shap_df.head())

    # Test a classification model
    cmodel = Model("wine-classification")
    c_importance_data = shap_feature_importance(cmodel)
    for feature, importance in c_importance_data:
        print(f"  {feature}: {importance:.4f}")

    # Get instance explanation data
    print("\n=== Instance Explanation Data Example (Classification) ===")
    shap_df_dict = shap_values_data(cmodel)
    for class_name, df in shap_df_dict.items():
        print(f"\nClass: {class_name}")
        print(df.head())

    # Test passing in shap data
    print("\n=== Passing in SHAP Data (Regression) ===")
    importance_data = shap_feature_importance(model, shap_data=shap_df)
    print("SHAP feature importance:")
    for feature, importance in importance_data:
        print(f"  {feature}: {importance:.4f}")

    # Test passing in shap data for classification
    print("\n=== Passing in SHAP Data (Classification) ===")
    importance_data = shap_feature_importance(cmodel, shap_data=shap_df_dict)
    print("SHAP feature importance:")
    for feature, importance in importance_data:
        print(f"  {feature}: {importance:.4f}")
