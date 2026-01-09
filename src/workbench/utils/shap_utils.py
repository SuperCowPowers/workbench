"""Model Utilities for Workbench models with explanation capabilities"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union

# Workbench Imports
from workbench.utils.xgboost_model_utils import xgboost_model_from_s3
from workbench.utils.model_utils import load_category_mappings_from_s3
from workbench.utils.pandas_utils import convert_categorical_types
from workbench.model_script_utils.model_script_utils import decompress_features

# Set up the log
log = logging.getLogger("workbench")


def shap_feature_importance(workbench_model) -> Optional[List[Tuple[str, float]]]:
    """
    Get feature importance data based on SHAP values from a Workbench Model.
    Works with both regression and multi-class classification models.

    Args:
        workbench_model: Workbench Model object

    Returns:
        List of tuples (feature, importance) sorted by importance (descending)
        or None if there was an error
    """
    # Compute the SHAP values
    log.important("Calculating SHAP values...")
    shap_data, feature_df = shap_values_data(workbench_model)
    if shap_data is None:
        log.error("No SHAP data found.")
        return None

    # Okay, we need the feature list (first column is ID, last column is bias)
    if isinstance(shap_data, dict):
        first_class_df = next(iter(shap_data.values()))
        features = first_class_df.columns.tolist()[1:-1]
    else:
        features = shap_data.columns.tolist()[1:-1]

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
    return sorted_importance


def shap_values_data(
    workbench_model, sample_df: pd.DataFrame = None
) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], pd.DataFrame]:
    """
    Get SHAP explanation data for all instances in the training data.
    Handles both regression/binary classification and multi-class models.

    Args:
        workbench_model: Workbench Model object
        sample_df: Optional DataFrame to sample from (default: None)

    Returns:
        For regression/binary: DataFrame with SHAP values, one row per instance, columns are features
        OR
        For multi-class: Dictionary of DataFrames, one per class, each with SHAP values
        AND
        Feature DataFrame with all features used for explanation

    Note:
        The ID column is always included as the first column of each DataFrame.
    """
    # Get all shap data from internal function
    features, shap_values, feature_df, ids = _calculate_shap_values(workbench_model, sample_df=sample_df)
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

        # Create a DataFrame for EACH class
        result_dict = {}
        for idx, label in enumerate(class_labels):
            class_df = pd.DataFrame(shap_values[:, idx, :], columns=features)
            class_df.insert(0, ids.name, ids.reset_index(drop=True))
            result_dict[label] = class_df
        return result_dict, feature_df

    # For regression or binary classification models (single class)
    else:
        # Extract the single class from the 3D array
        single_class_values = shap_values[:, 0, :]
        result_df = pd.DataFrame(single_class_values, columns=features)
        result_df.insert(0, ids.name, ids.reset_index(drop=True))
        return result_df, feature_df


def _calculate_shap_values(workbench_model, sample_df: pd.DataFrame = None):
    """
    Internal function to calculate SHAP values for Workbench Models.
    Handles both regression and multi-class classification models.
    Supports categorical features if mappings are available.

    Args:
        workbench_model: Workbench Model object
        sample_df: Optional DataFrame to sample from (default: None)

    Note:
        If you set sample_df, the model must have 'shap_importance' already computed

    Returns:
        - list of feature names
        - raw shap values
        - feature data used for explanation
        - ids of the input data
        or (None, None, None, None) if there was an error
    """
    import xgboost as xgb
    from workbench.api import FeatureSet

    # Get input data for the model
    fs = FeatureSet(workbench_model.get_input())
    features = workbench_model.features()

    # Did we get a sample DataFrame?
    if sample_df is not None:
        log.info("Sampling rows with sample dataframe...")
        X = sample_df[features]
        ids = sample_df[fs.id_column]

    # Full run using all the columns and all the rows
    else:
        df = fs.pull_dataframe()
        X = df[features]
        ids = df[fs.id_column]

    # Get the XGBoost model from the Workbench Model
    model_artifact_uri = workbench_model.model_data_url()
    xgb_model = xgboost_model_from_s3(model_artifact_uri)
    if xgb_model is None:
        log.error("No XGBoost model found in the artifact.")
        return None, None, None, None

    # Get the booster (SHAP requires the booster, not the sklearn wrapper)
    if hasattr(xgb_model, "get_booster"):
        # Full sklearn model - extract the booster
        booster = xgb_model.get_booster()
    else:
        # Already a booster
        booster = xgb_model

    # Load category mappings if available
    category_mappings = load_category_mappings_from_s3(model_artifact_uri)

    # Apply categorical conversions if mappings exist
    if category_mappings:
        log.info("Category mappings found. Applying categorical conversions.")
        X = convert_categorical_types(X, category_mappings)

    # Check if we have compressed features to decompress
    compressed_features = fs.get_compressed_features()
    if compressed_features:
        log.info("Decompressing compressed features...")
        X, features = decompress_features(X, features, compressed_features)

    # Create a DMatrix with categorical support
    dmatrix = xgb.DMatrix(X, enable_categorical=True)

    # Use XGBoost's built-in SHAP calculation (booster method, not sklearn)
    shap_values = booster.predict(dmatrix, pred_contribs=True, strict_shape=True)
    features_with_bias = features + ["bias"]

    # Now we need to subset the columns based on top 10 SHAP values
    if sample_df is not None:
        # Get just the feature names from your top_shap
        top_shap = [f[0] for f in workbench_model.shap_importance()][:10]

        # Find indices of these top features in original features list
        top_indices = [features.index(feat) for feat in top_shap]

        # Filter shap_values (-1 is to get the bias term)
        shap_values = shap_values[:, :, top_indices + [-1]]

        # Update features list to match
        features_with_bias = [features[i] for i in top_indices] + ["bias"]

    # For the feature dataframe we're going to add the id column as the first column
    feature_df = pd.concat([ids.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Return the feature names, SHAP values, feature dataframe, and IDs
    return features_with_bias, shap_values, feature_df, ids


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model, FeatureSet

    # Set pandas display options
    pd.options.display.max_columns = 20
    pd.options.display.max_colwidth = 200
    pd.options.display.width = 1400

    # Test a regression model
    model = Model("test-regression")
    model = Model("aqsol-fingerprints")
    model.compute_shap_values()

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
    shap_df, feature_df = shap_values_data(model)
    print(shap_df.head())

    # Test a classification model
    cmodel = Model("wine-classification")
    c_importance_data = shap_feature_importance(cmodel)
    for feature, importance in c_importance_data:
        print(f"  {feature}: {importance:.4f}")

    # Get instance explanation data
    print("\n=== Instance Explanation Data Example (Classification) ===")
    shap_df_dict, feature_df = shap_values_data(cmodel)
    for class_name, df in shap_df_dict.items():
        print(f"\nClass: {class_name}")
        print(df.head())

    # Test SHAP values data with sampling (regression)
    model = Model("abalone-regression")
    my_sample_df = FeatureSet(model.get_input()).pull_dataframe().sample(1000)
    print("\n=== SHAP Values Data with Sampling (regression) ===")
    shap_df_sample, feature_df = shap_values_data(model, sample_df=my_sample_df)
    print(shap_df_sample.head())

    # Test SHAP values data with sampling (classification)
    model = Model("wine-classification")
    my_sample_df = FeatureSet(model.get_input()).pull_dataframe().sample(100)
    print("\n=== SHAP Values Data with Sampling (classification) ===")
    shap_df_sample, feature_df = shap_values_data(model, sample_df=my_sample_df)
    for class_name, df in shap_df_sample.items():
        print(f"\nClass: {class_name}")
        print(df.head())
