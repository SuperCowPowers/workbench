import shap
import pandas as pd
import awswrangler as wr

# SageWorks Imports
from sageworks.utils.extract_model_artifact import ExtractModelArtifact

# SageWorks logging
import logging

log = logging.getLogger("sageworks")


def generate_shap_values(
    endpoint_name: str, model_type: str, pred_results_df: pd.DataFrame, inference_capture_path: str
):
    """Compute the SHAP values for this Model associated with this Endpoint

    Args:
        endpoint_name (str): Name of the endpoint to extract the model artifact from
        model_type (str): Type of model (classifier or regressor)
        pred_results_df (pd.DataFrame): DataFrame with the prediction results
        inference_capture_path (str): S3 Path to the Inference Capture Folder

    Notes:
        Writes the SHAP values to the S3 Inference Capture Folder
    """
    # Grab the model artifact from AWS
    model_artifact = ExtractModelArtifact(endpoint_name).get_model_artifact()

    # Do we have a model artifact?
    if model_artifact is None:
        log.error(f"Could not find model artifact for {endpoint_name}")
        return

    # Get the exact features used to train the model
    model_features = model_artifact.feature_names_in_
    X_pred = pred_results_df[model_features]

    # Compute the SHAP values
    try:
        # Note: For Tree-based models like decision trees, random forests, XGBoost, LightGBM,
        explainer = shap.TreeExplainer(model_artifact)
        shap_vals = explainer.shap_values(X_pred)

        # Multiple shap vals CSV for classifiers
        if model_type == "classifier":
            # Need a separate shapley values CSV for each class
            for i, class_shap_vals in enumerate(shap_vals):
                df_shap = pd.DataFrame(class_shap_vals, columns=X_pred.columns)

                # Write shap vals to S3 Model Inference Folder
                shap_file_path = f"{inference_capture_path}/inference_shap_values_class_{i}.csv"
                log.info(f"Writing SHAP values to {shap_file_path}")
                wr.s3.to_csv(df_shap, shap_file_path, index=False)

        # Single shap vals CSV for regressors
        if model_type == "regressor":
            # Format shap values into single dataframe
            df_shap = pd.DataFrame(shap_vals, columns=X_pred.columns)

            # Write shap vals to S3 Model Inference Folder
            log.info(f"Writing SHAP values to {inference_capture_path}/inference_shap_values.csv")
            wr.s3.to_csv(df_shap, f"{inference_capture_path}/inference_shap_values.csv", index=False)

    except Exception as e:
        log.error(f"Error computing SHAP values: {e}")


if __name__ == "__main__":
    """Exercise the Shapley Values Method"""
    from sageworks.api import Endpoint

    # Grab an endpoint
    endpoint_name = "abalone-regression-end"
    end = Endpoint(endpoint_name)

    # Define the input parameters
    model_type = end.model_type()
    pred_results_df = end.auto_inference()
    inference_capture_path = f"{end.endpoint_inference_path}/test_capture"

    # Generate the SHAP values
    generate_shap_values(endpoint_name, model_type, pred_results_df, inference_capture_path)
