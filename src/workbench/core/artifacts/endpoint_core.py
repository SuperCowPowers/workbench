"""EndpointCore: Workbench EndpointCore Class"""

import time
from datetime import datetime
import botocore
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from io import StringIO
import awswrangler as wr
from typing import Union, Optional
import hashlib

# Model Performance Scores
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    mean_squared_error,
)
from sklearn.preprocessing import OneHotEncoder

# SageMaker Imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts import FeatureSetCore, ModelCore, ModelType
from workbench.utils.endpoint_metrics import EndpointMetrics

# from workbench.utils.shapley_values import generate_shap_values
from workbench.utils.fast_inference import fast_inference
from workbench.utils.cache import Cache
from workbench.utils.s3_utils import compute_s3_object_hash


class EndpointCore(Artifact):
    """EndpointCore: Workbench EndpointCore Class

    Common Usage:
        ```python
        my_endpoint = EndpointCore(endpoint_uuid)
        prediction_df = my_endpoint.predict(test_df)
        metrics = my_endpoint.regression_metrics(target_column, prediction_df)
        for metric, value in metrics.items():
            print(f"{metric}: {value:0.3f}")
        ```
    """

    def __init__(self, endpoint_uuid, **kwargs):
        """EndpointCore Initialization

        Args:
            endpoint_uuid (str): Name of Endpoint in Workbench
        """

        # Make sure the endpoint_uuid is a valid name
        self.is_name_valid(endpoint_uuid, delimiter="-", lower_case=False)

        # Call SuperClass Initialization
        super().__init__(endpoint_uuid, **kwargs)

        # Grab an Cloud Metadata object and pull information for Endpoints
        self.endpoint_name = endpoint_uuid
        self.endpoint_meta = self.meta.endpoint(self.endpoint_name)

        # Sanity check that we found the endpoint
        if self.endpoint_meta is None:
            self.log.important(f"Could not find endpoint {self.uuid} within current visibility scope")
            return

        # Sanity check the Endpoint state
        if self.endpoint_meta["EndpointStatus"] == "Failed":
            self.log.critical(f"Endpoint {self.uuid} is in a failed state")
            reason = self.endpoint_meta["FailureReason"]
            self.log.critical(f"Failure Reason: {reason}")
            self.log.critical("Please delete this endpoint and re-deploy...")

        # Set the Inference, Capture, and Monitoring S3 Paths
        self.endpoint_inference_path = self.endpoints_s3_path + "/inference/" + self.uuid
        self.endpoint_data_capture_path = self.endpoints_s3_path + "/data_capture/" + self.uuid
        self.endpoint_monitoring_path = self.endpoints_s3_path + "/monitoring/" + self.uuid

        # Set the Model Name
        self.model_name = self.get_input()

        # This is for endpoint error handling later
        self.endpoint_return_columns = None

        # We temporary cache the endpoint metrics
        self.temp_storage = Cache(prefix="temp_storage", expire=300)  # 5 minutes

        # Call SuperClass Post Initialization
        super().__post_init__()

        # All done
        self.log.info(f"EndpointCore Initialized: {self.endpoint_name}")

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self.endpoint_meta = self.meta.endpoint(self.endpoint_name)

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.endpoint_meta is None:
            self.log.debug(f"Endpoint {self.endpoint_name} not found in AWS Metadata")
            return False
        return True

    def health_check(self) -> list[str]:
        """Perform a health check on this model

        Returns:
            list[str]: List of health issues
        """
        if not self.ready():
            return ["needs_onboard"]

        # Call the base class health check
        health_issues = super().health_check()

        # Does this endpoint have a config?
        # Note: This is not an authoritative check, so improve later
        if self.endpoint_meta.get("ProductionVariants") is None:
            health_issues.append("no_config")

        # We're going to check for 5xx errors and no activity
        endpoint_metrics = self.endpoint_metrics()

        # Check if we have metrics
        if endpoint_metrics is None:
            health_issues.append("unknown_error")
            return health_issues

        # Check for 5xx errors
        num_errors = endpoint_metrics["Invocation5XXErrors"].sum()
        if num_errors > 5:
            health_issues.append("5xx_errors")
        elif num_errors > 0:
            health_issues.append("5xx_errors_min")
        else:
            self.remove_health_tag("5xx_errors")
            self.remove_health_tag("5xx_errors_min")

        # Check for Endpoint activity
        num_invocations = endpoint_metrics["Invocations"].sum()
        if num_invocations == 0:
            health_issues.append("no_activity")
        else:
            self.remove_health_tag("no_activity")
        return health_issues

    def is_serverless(self) -> bool:
        """Check if the current endpoint is serverless.

        Returns:
            bool: True if the endpoint is serverless, False otherwise.
        """
        return "Serverless" in self.endpoint_meta["InstanceType"]

    def add_data_capture(self):
        """Add data capture to the endpoint"""
        self.get_monitor().add_data_capture()

    def get_monitor(self):
        """Get the MonitorCore class for this endpoint"""
        from workbench.core.artifacts.monitor_core import MonitorCore

        return MonitorCore(self.endpoint_name)

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        return 0.0

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.endpoint_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.endpoint_meta["EndpointArn"]

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.endpoint_meta["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.endpoint_meta["LastModifiedTime"]

    def model_data_url(self) -> Optional[str]:
        """Return the model data URL for this endpoint

        Returns:
            Optional[str]: The model data URL for this endpoint
        """
        from workbench.utils.endpoint_utils import internal_model_data_url  # Avoid circular import

        return internal_model_data_url(self.endpoint_config_name(), self.boto3_session)

    def hash(self) -> Optional[str]:
        """Return the hash for the internal model used by this endpoint

        Returns:
            Optional[str]: The hash for the internal model used by this endpoint
        """
        model_url = self.model_data_url()
        return compute_s3_object_hash(model_url, self.boto3_session)

    @property
    def instance_type(self) -> str:
        """Return the instance type for this endpoint"""
        return self.endpoint_meta["InstanceType"]

    def endpoint_metrics(self) -> Union[pd.DataFrame, None]:
        """Return the metrics for this endpoint

        Returns:
            pd.DataFrame: DataFrame with the metrics for this endpoint (or None if no metrics)
        """

        # Do we have it cached?
        metrics_key = f"endpoint:{self.uuid}:endpoint_metrics"
        endpoint_metrics = self.temp_storage.get(metrics_key)
        if endpoint_metrics is not None:
            return endpoint_metrics

        # We don't have it cached so let's get it from CloudWatch
        if "ProductionVariants" not in self.endpoint_meta:
            return None
        self.log.important("Updating endpoint metrics...")
        variant = self.endpoint_meta["ProductionVariants"][0]["VariantName"]
        endpoint_metrics = EndpointMetrics().get_metrics(self.uuid, variant=variant)
        self.temp_storage.set(metrics_key, endpoint_metrics)
        return endpoint_metrics

    def details(self) -> dict:
        """Additional Details about this Endpoint

        Returns:
            dict(dict): A dictionary of details about this Endpoint
        """

        # Fill in all the details about this Endpoint
        details = self.summary()

        # Get details from our AWS Metadata
        details["status"] = self.endpoint_meta["EndpointStatus"]
        details["instance"] = self.endpoint_meta["InstanceType"]
        try:
            details["instance_count"] = self.endpoint_meta["ProductionVariants"][0]["CurrentInstanceCount"] or "-"
        except KeyError:
            details["instance_count"] = "-"
        if "ProductionVariants" in self.endpoint_meta:
            details["variant"] = self.endpoint_meta["ProductionVariants"][0]["VariantName"]
        else:
            details["variant"] = "-"

        # Add endpoint metrics from CloudWatch
        details["endpoint_metrics"] = self.endpoint_metrics()

        # Return the details
        return details

    def onboard(self, interactive: bool = False) -> bool:
        """This is a BLOCKING method that will onboard the Endpoint (make it ready)
        Args:
            interactive (bool, optional): If True, will prompt the user for information. (default: False)
        Returns:
            bool: True if the Endpoint is successfully onboarded, False otherwise
        """

        # Make sure our input is defined
        if self.get_input() == "unknown":
            if interactive:
                input_model = input("Input Model?: ")
            else:
                self.log.critical("Input Model is not defined!")
                return False
        else:
            input_model = self.get_input()

        # Now that we have the details, let's onboard the Endpoint with args
        return self.onboard_with_args(input_model)

    def onboard_with_args(self, input_model: str) -> bool:
        """Onboard the Endpoint with the given arguments

        Args:
            input_model (str): The input model for this endpoint
        Returns:
            bool: True if the Endpoint is successfully onboarded, False otherwise
        """
        # Set the status to onboarding
        self.set_status("onboarding")

        self.upsert_workbench_meta({"workbench_input": input_model})
        self.model_name = input_model

        # Remove the needs_onboard tag
        self.remove_health_tag("needs_onboard")
        self.set_status("ready")

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details()
        return True

    def auto_inference(self, capture: bool = False) -> pd.DataFrame:
        """Run inference on the endpoint using FeatureSet data

        Args:
            capture (bool, optional): Capture the inference results and metrics (default=False)
        """

        # Sanity Check that we have a model
        model = ModelCore(self.get_input())
        if not model.exists():
            self.log.error("No model found for this endpoint. Returning empty DataFrame.")
            return pd.DataFrame()

        # Now get the FeatureSet and make sure it exists
        fs = FeatureSetCore(model.get_input())
        if not fs.exists():
            self.log.error("No FeatureSet found for this endpoint. Returning empty DataFrame.")
            return pd.DataFrame()

        # Grab the evaluation data from the FeatureSet
        table = fs.view("training").table
        eval_df = fs.query(f'SELECT * FROM "{table}" where training = FALSE')
        capture_uuid = "auto_inference" if capture else None
        return self.inference(eval_df, capture_uuid, id_column=fs.id_column)

    def inference(self, eval_df: pd.DataFrame, capture_uuid: str = None, id_column: str = None) -> pd.DataFrame:
        """Run inference and compute performance metrics with optional capture

        Args:
            eval_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
            capture_uuid (str, optional): UUID of the inference capture (default=None)
            id_column (str, optional): Name of the ID column (default=None)

        Returns:
            pd.DataFrame: DataFrame with the inference results

        Note:
            If capture=True inference/performance metrics are written to S3 Endpoint Inference Folder
        """

        # Check if this is a 'floating endpoint' (no model)
        if self.get_input() == "unknown":
            self.log.important("No model associated with this endpoint, running 'no frills' inference...")
            return self.fast_inference(eval_df)

        # Run predictions on the evaluation data
        prediction_df = self._predict(eval_df)
        if prediction_df.empty:
            self.log.warning("No predictions were made. Returning empty DataFrame.")
            return prediction_df

        # Get the target column
        model = ModelCore(self.model_name)
        target_column = model.target()

        # Sanity Check that the target column is present
        if target_column and (target_column not in prediction_df.columns):
            self.log.important(f"Target Column {target_column} not found in prediction_df!")
            self.log.important("In order to compute metrics, the target column must be present!")
            return prediction_df

        # Compute the standard performance metrics for this model
        model_type = model.model_type
        if model_type in [ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]:
            prediction_df = self.residuals(target_column, prediction_df)
            metrics = self.regression_metrics(target_column, prediction_df)
        elif model_type == ModelType.CLASSIFIER:
            metrics = self.classification_metrics(target_column, prediction_df)
        else:
            # For other model types, we don't compute metrics
            self.log.info(f"Model Type: {model_type} doesn't have metrics...")
            metrics = pd.DataFrame()

        # Print out the metrics
        if not metrics.empty:
            print(f"Performance Metrics for {self.model_name} on {self.uuid}")
            print(metrics.head())

            # Capture the inference results and metrics
            if capture_uuid is not None:
                description = capture_uuid.replace("_", " ").title()
                features = model.features()
                self._capture_inference_results(
                    capture_uuid, prediction_df, target_column, model_type, metrics, description, features, id_column
                )

        # Return the prediction DataFrame
        return prediction_df

    def fast_inference(self, eval_df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            threads (int): The number of threads to use (default: 4)

        Returns:
            pd.DataFrame: The DataFrame with predictions

        Note:
            There's no sanity checks or error handling... just FAST Inference!
        """
        return fast_inference(self.uuid, eval_df, self.sm_session, threads=threads)

    def _predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Run prediction on the given observations in the given DataFrame
        Args:
            eval_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
        Returns:
            pd.DataFrame: Return the DataFrame with additional columns, prediction and any _proba columns
        """

        # Sanity check: Does the DataFrame have 0 rows?
        if eval_df.empty:
            self.log.warning("Evaluation DataFrame has 0 rows. No predictions to run.")
            return pd.DataFrame(columns=eval_df.columns)  # Return empty DataFrame with same structure

        # Sanity check: Does the Model have Features?
        features = ModelCore(self.model_name).features()
        if not features:
            self.log.warning("Model does not have features defined, using all columns in the DataFrame")
        else:
            # Sanity check: Does the DataFrame have the required features?
            df_columns_lower = set(col.lower() for col in eval_df.columns)
            features_lower = set(feature.lower() for feature in features)

            # Check if the features are a subset of the DataFrame columns (case-insensitive)
            if not features_lower.issubset(df_columns_lower):
                missing_features = features_lower - df_columns_lower
                raise ValueError(f"DataFrame does not contain required features: {missing_features}")

        # Create our Endpoint Predictor Class
        predictor = Predictor(
            self.endpoint_name,
            sagemaker_session=self.sm_session,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
        )

        # Now split up the dataframe into 100 row chunks, send those chunks to our
        # endpoint (with error handling) and stitch all the chunks back together
        df_list = []
        total_rows = len(eval_df)
        for index in range(0, len(eval_df), 100):
            self.log.info(f"Processing {index}:{min(index+100, total_rows)} out of {total_rows} rows...")

            # Compute partial DataFrames, add them to a list, and concatenate at the end
            partial_df = self._endpoint_error_handling(predictor, eval_df[index : index + 100])
            df_list.append(partial_df)

        # Concatenate the dataframes
        combined_df = pd.concat(df_list, ignore_index=True)

        # Some endpoints will put in "N/A" values (for CSV serialization)
        # We need to convert these to NaN and the run the conversions below
        # Report on the number of N/A values in each column in the DataFrame
        # For any cound above 0 list the column name and the number of N/A values
        na_counts = combined_df.isin(["N/A"]).sum()
        for column, count in na_counts.items():
            if count > 0:
                self.log.warning(f"{column} has {count} N/A values, converting to NaN")
        pd.set_option("future.no_silent_downcasting", True)
        combined_df = combined_df.replace("N/A", float("nan"))

        # Convert data to numeric
        # Note: Since we're using CSV serializers numeric columns often get changed to generic 'object' types

        # Hard Conversion
        # Note: We explicitly catch exceptions for columns that cannot be converted to numeric
        converted_df = combined_df.copy()
        for column in combined_df.columns:
            try:
                converted_df[column] = pd.to_numeric(combined_df[column])
            except ValueError:
                # If a ValueError is raised, the column cannot be converted to numeric, so we keep it as is
                pass
            except TypeError:
                # This typically means a duplicated column name, so confirm duplicate (more than 1) and log it
                column_count = (converted_df.columns == column).sum()
                self.log.critical(f"{column} occurs {column_count} times in the DataFrame.")
                pass

        # Soft Conversion
        # Convert columns to the best possible dtype that supports the pd.NA missing value.
        converted_df = converted_df.convert_dtypes()

        # Convert pd.NA placeholders to pd.NA
        # Note: CSV serialization converts pd.NA to blank strings, so we have to put in placeholders
        converted_df.replace("__NA__", pd.NA, inplace=True)

        # Check for True/False values in the string columns
        for column in converted_df.select_dtypes(include=["string"]).columns:
            if converted_df[column].str.lower().isin(["true", "false"]).all():
                converted_df[column] = converted_df[column].str.lower().map({"true": True, "false": False})

        # Return the Dataframe
        return converted_df

    def _endpoint_error_handling(self, predictor, feature_df):
        """Internal: Handles errors, retries, and binary search for problematic rows."""

        # Sanity check: Does the DataFrame have 0 rows?
        if feature_df.empty:
            self.log.warning("DataFrame has 0 rows. No predictions to run.")
            return pd.DataFrame(columns=feature_df.columns)

        # Convert DataFrame into a CSV buffer
        csv_buffer = StringIO()
        feature_df.to_csv(csv_buffer, index=False)

        try:
            # Send CSV buffer to the predictor and process results
            results = predictor.predict(csv_buffer.getvalue())
            results_df = pd.DataFrame.from_records(results[1:], columns=results[0])
            self.endpoint_return_columns = results_df.columns.tolist()
            return results_df

        except botocore.exceptions.ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code == "ModelNotReadyException":
                self.log.error(f"Error {error_code}: {err.response.get('Message', 'No message')}")
                self.log.error("Model not ready. Sleeping and retrying...")
                time.sleep(60)
                return self._endpoint_error_handling(predictor, feature_df)

            elif error_code == "ModelError":
                self.log.warning("Model error. Bisecting the DataFrame and retrying...")

                # Base case: If there is only one row, we can't binary search further
                if len(feature_df) == 1:
                    if not self.endpoint_return_columns:
                        raise

                    # Fill the row with NaNs for endpoint_return_columns
                    self.log.warning(f"Endpoint Inference failed on :{feature_df}")
                    # return pd.DataFrame(columns=feature_df.columns)  # Empty DataFrame with same structure
                    return self._fill_with_nans(feature_df)

                # Binary search to find the problematic row(s)
                mid_point = len(feature_df) // 2
                self.log.info(f"Bisect DataFrame: 0 -> {mid_point} and {mid_point} -> {len(feature_df)}")
                first_half = self._endpoint_error_handling(predictor, feature_df.iloc[:mid_point])
                second_half = self._endpoint_error_handling(predictor, feature_df.iloc[mid_point:])
                return pd.concat([first_half, second_half], ignore_index=True)

            else:
                # Unknown ClientError, raise the exception
                self.log.critical(f"Unexpected ClientError: {err}")
                raise

        except Exception as err:
            self.log.critical(f"Unexpected general error: {err}")
            raise

    def _fill_with_nans(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Fill a single-row DataFrame with NaNs for inference columns, keeping original feature data."""

        # Create a single-row DataFrame with NaNs, ensuring dtype=object to prevent type downcasting
        one_row_df_with_nans = pd.DataFrame({col: [np.NaN] for col in self.endpoint_return_columns}, dtype=object)

        # Check if feature_df is not empty and has at least one row
        if not feature_df.empty:
            # Copy values from the input DataFrame for overlapping columns
            for column in feature_df.columns:
                # Use .iloc[0] to access the first row by position, regardless of the index
                one_row_df_with_nans.at[0, column] = feature_df.iloc[0][column]

        return one_row_df_with_nans

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame, hash_length: int = 8):
        # Internal: Compute a data hash for the dataframe
        df = df.copy()
        df = df.sort_values(by=sorted(df.columns.tolist()))
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        combined = row_hashes.values.tobytes()
        return hashlib.md5(combined).hexdigest()[:hash_length]

    def _capture_inference_results(
        self,
        capture_uuid: str,
        pred_results_df: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        metrics: pd.DataFrame,
        description: str,
        features: list,
        id_column: str = None,
    ):
        """Internal: Capture the inference results and metrics to S3

        Args:
            capture_uuid (str): UUID of the inference capture
            pred_results_df (pd.DataFrame): DataFrame with the prediction results
            target_column (str): Name of the target column
            model_type (ModelType): Type of the model (e.g. REGRESSOR, CLASSIFIER)
            metrics (pd.DataFrame): DataFrame with the performance metrics
            description (str): Description of the inference results
            features (list): List of features to include in the inference results
            id_column (str, optional): Name of the ID column (default=None)
        """

        # Compute a dataframe hash (just use the last 8)
        data_hash = self._hash_dataframe(pred_results_df[features])

        # Metadata for the model inference
        inference_meta = {
            "name": capture_uuid,
            "data_hash": data_hash,
            "num_rows": len(pred_results_df),
            "description": description,
        }

        # Create the S3 Path for the Inference Capture
        inference_capture_path = f"{self.endpoint_inference_path}/{capture_uuid}"

        # Write the metadata dictionary and metrics to our S3 Model Inference Folder
        wr.s3.to_json(
            pd.DataFrame([inference_meta]),
            f"{inference_capture_path}/inference_meta.json",
            index=False,
        )
        self.log.info(f"Writing metrics to {inference_capture_path}/inference_metrics.csv")
        wr.s3.to_csv(metrics, f"{inference_capture_path}/inference_metrics.csv", index=False)

        # Grab the target column, prediction column, any _proba columns, and the ID column (if present)
        prediction_col = "prediction" if "prediction" in pred_results_df.columns else "predictions"
        output_columns = [target_column, prediction_col]

        # Add any _proba columns to the output columns
        output_columns += [col for col in pred_results_df.columns if col.endswith("_proba")]

        # Add any quantile columns to the output columns
        output_columns += [col for col in pred_results_df.columns if col.startswith("q_") or col.startswith("qr_")]

        # Add the ID column
        if id_column and id_column in pred_results_df.columns:
            output_columns.append(id_column)

        # Write the predictions to our S3 Model Inference Folder
        self.log.info(f"Writing predictions to {inference_capture_path}/inference_predictions.csv")
        subset_df = pred_results_df[output_columns]
        wr.s3.to_csv(subset_df, f"{inference_capture_path}/inference_predictions.csv", index=False)

        # CLASSIFIER: Write the confusion matrix to our S3 Model Inference Folder
        if model_type == ModelType.CLASSIFIER:
            conf_mtx = self.generate_confusion_matrix(target_column, pred_results_df)
            self.log.info(f"Writing confusion matrix to {inference_capture_path}/inference_cm.csv")
            # Note: Unlike other dataframes here, we want to write the index (labels) to the CSV
            wr.s3.to_csv(conf_mtx, f"{inference_capture_path}/inference_cm.csv", index=True)

        # Generate SHAP values for our Prediction Dataframe
        # generate_shap_values(self.endpoint_name, model_type.value, pred_results_df, inference_capture_path)

        # Now recompute the details for our Model
        self.log.important(f"Recomputing Details for {self.model_name} to show latest Inference Results...")
        model = ModelCore(self.model_name)
        model._load_inference_metrics(capture_uuid)
        model.details()

        # Recompute the details so that inference model metrics are updated
        self.log.important(f"Recomputing Details for {self.uuid} to show latest Inference Results...")
        self.details()

    def regression_metrics(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """

        # Sanity Check the prediction DataFrame
        if prediction_df.empty:
            self.log.warning("No predictions were made. Returning empty DataFrame.")
            return pd.DataFrame()

        # Compute the metrics
        y_true = prediction_df[target_column]
        prediction_col = "prediction" if "prediction" in prediction_df.columns else "predictions"
        y_pred = prediction_df[prediction_col]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        # Mean Absolute Percentage Error
        mape = np.mean(np.where(y_true != 0, np.abs((y_true - y_pred) / y_true), np.abs(y_true - y_pred))) * 100
        # Median Absolute Error
        medae = median_absolute_error(y_true, y_pred)

        # Organize and return the metrics
        metrics = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
            "MAPE": round(mape, 3),
            "MedAE": round(medae, 3),
            "NumRows": len(prediction_df),
        }
        return pd.DataFrame.from_records([metrics])

    def residuals(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Add the residuals to the prediction DataFrame
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with two new columns called 'residuals' and 'residuals_abs'
        """

        # Compute the residuals
        y_true = prediction_df[target_column]
        prediction_col = "prediction" if "prediction" in prediction_df.columns else "predictions"
        y_pred = prediction_df[prediction_col]

        # Check for classification scenario
        if not pd.api.types.is_numeric_dtype(y_true) or not pd.api.types.is_numeric_dtype(y_pred):
            self.log.warning("Target and Prediction columns are not numeric. Computing 'diffs'...")
            prediction_df["residuals"] = (y_true != y_pred).astype(int)
            prediction_df["residuals_abs"] = prediction_df["residuals"]
        else:
            # Compute numeric residuals for regression
            prediction_df["residuals"] = y_true - y_pred
            prediction_df["residuals_abs"] = np.abs(prediction_df["residuals"])

        return prediction_df

    @staticmethod
    def validate_proba_columns(prediction_df: pd.DataFrame, class_labels: list, guessing: bool = False):
        """Ensure probability columns are correctly aligned with class labels

        Args:
            prediction_df (pd.DataFrame): DataFrame with the prediction results
            class_labels (list): List of class labels
            guessing (bool, optional): Whether we're guessing the class labels. Defaults to False.
        """
        proba_columns = [col.replace("_proba", "") for col in prediction_df.columns if col.endswith("_proba")]

        if sorted(class_labels) != sorted(proba_columns):
            if guessing:
                raise ValueError(f"_proba columns {proba_columns} != GUESSED class_labels {class_labels}!")
            else:
                raise ValueError(f"_proba columns {proba_columns} != class_labels {class_labels}!")

    def classification_metrics(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint

        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results

        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """
        # Get the class labels from the model
        class_labels = ModelCore(self.model_name).class_labels()
        if class_labels is None:
            self.log.warning(
                "Class labels not found in the model. Guessing class labels from the prediction DataFrame."
            )
            class_labels = prediction_df[target_column].unique().tolist()
            self.validate_proba_columns(prediction_df, class_labels, guessing=True)
        else:
            self.validate_proba_columns(prediction_df, class_labels)

        # Calculate precision, recall, fscore, and support, handling zero division
        prediction_col = "prediction" if "prediction" in prediction_df.columns else "predictions"
        scores = precision_recall_fscore_support(
            prediction_df[target_column],
            prediction_df[prediction_col],
            average=None,
            labels=class_labels,
            zero_division=0,
        )

        # Identify the probability columns and keep them as a Pandas DataFrame
        proba_columns = [f"{label}_proba" for label in class_labels]
        y_score = prediction_df[proba_columns]

        # One-hot encode the true labels using all class labels (fit with class_labels)
        encoder = OneHotEncoder(categories=[class_labels], sparse_output=False)
        y_true = encoder.fit_transform(prediction_df[[target_column]])

        # Calculate ROC AUC per label and handle exceptions for missing classes
        roc_auc_per_label = []
        for i, label in enumerate(class_labels):
            try:
                roc_auc = roc_auc_score(y_true[:, i], y_score.iloc[:, i])
            except ValueError as e:
                self.log.warning(f"ROC AUC calculation failed for label {label}.")
                self.log.warning(f"{str(e)}")
                roc_auc = 0.0
            roc_auc_per_label.append(roc_auc)

        # Put the scores into a DataFrame
        score_df = pd.DataFrame(
            {
                target_column: class_labels,
                "precision": scores[0],
                "recall": scores[1],
                "fscore": scores[2],
                "roc_auc": roc_auc_per_label,
                "support": scores[3],
            }
        )
        return score_df

    def generate_confusion_matrix(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the confusion matrix for this Endpoint
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the confusion matrix
        """

        y_true = prediction_df[target_column]
        prediction_col = "prediction" if "prediction" in prediction_df.columns else "predictions"
        y_pred = prediction_df[prediction_col]

        # Check if our model has class labels, if not we'll use the unique labels in the prediction
        class_labels = ModelCore(self.model_name).class_labels()
        if class_labels is None:
            class_labels = sorted(list(set(y_true) | set(y_pred)))

        # Compute the confusion matrix (sklearn confusion_matrix)
        conf_mtx = confusion_matrix(y_true, y_pred, labels=class_labels)

        # Create a DataFrame
        conf_mtx_df = pd.DataFrame(conf_mtx, index=class_labels, columns=class_labels)
        conf_mtx_df.index.name = "labels"

        # Check if our model has class labels. If so make the index and columns ordered
        model_class_labels = ModelCore(self.model_name).class_labels()
        if model_class_labels:
            self.log.important("Reordering the confusion matrix based on model class labels...")
            conf_mtx_df.index = pd.Categorical(conf_mtx_df.index, categories=model_class_labels, ordered=True)
            conf_mtx_df.columns = pd.Categorical(conf_mtx_df.columns, categories=model_class_labels, ordered=True)
            conf_mtx_df = conf_mtx_df.sort_index().sort_index(axis=1)
        return conf_mtx_df

    def endpoint_config_name(self) -> str:
        # Grab the Endpoint Config Name from the AWS
        details = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        return details["EndpointConfigName"]

    def set_input(self, input: str, force=False):
        """Override: Set the input data for this artifact

        Args:
            input (str): Name of input for this artifact
            force (bool, optional): Force the input to be set. Defaults to False.
        Note:
            We're going to not allow this to be used for Models
        """
        if not force:
            self.log.warning(f"Endpoint {self.uuid}: Does not allow manual override of the input!")
            return

        # Okay we're going to allow this to be set
        self.log.important(f"{self.uuid}: Setting input to {input}...")
        self.log.important("Be careful with this! It breaks automatic provenance of the artifact!")
        self.upsert_workbench_meta({"workbench_input": input})

    def delete(self):
        """ "Delete an existing Endpoint: Underlying Models, Configuration, and Endpoint"""
        if not self.exists():
            self.log.warning(f"Trying to delete an Model that doesn't exist: {self.uuid}")

        # Call the Class Method to delete the FeatureSet
        EndpointCore.managed_delete(endpoint_name=self.uuid)

    @classmethod
    def managed_delete(cls, endpoint_name: str):
        """Delete the Endpoint and associated resources if it exists"""

        # Check if the endpoint exists
        try:
            endpoint_info = cls.sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            if e.response["Error"]["Code"] in ["ValidationException", "ResourceNotFound"]:
                cls.log.info(f"Endpoint {endpoint_name} not found!")
                return
            raise  # Re-raise unexpected errors

        # Delete underlying models (Endpoints store/use models internally)
        cls.delete_endpoint_models(endpoint_name)

        # Get Endpoint Config Name and delete if exists
        endpoint_config_name = endpoint_info["EndpointConfigName"]
        try:
            cls.log.info(f"Deleting Endpoint Config {endpoint_config_name}...")
            cls.sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        except ClientError:
            cls.log.info(f"Endpoint Config {endpoint_config_name} not found...")

        # Delete any monitoring schedules associated with the endpoint
        monitoring_schedules = cls.sm_client.list_monitoring_schedules(EndpointName=endpoint_name)[
            "MonitoringScheduleSummaries"
        ]
        for schedule in monitoring_schedules:
            cls.log.info(f"Deleting Monitoring Schedule {schedule['MonitoringScheduleName']}...")
            cls.sm_client.delete_monitoring_schedule(MonitoringScheduleName=schedule["MonitoringScheduleName"])

        # Delete related S3 artifacts (inference, data capture, monitoring)
        endpoint_inference_path = cls.endpoints_s3_path + "/inference/" + endpoint_name
        endpoint_data_capture_path = cls.endpoints_s3_path + "/data_capture/" + endpoint_name
        endpoint_monitoring_path = cls.endpoints_s3_path + "/monitoring/" + endpoint_name
        for s3_path in [endpoint_inference_path, endpoint_data_capture_path, endpoint_monitoring_path]:
            s3_path = f"{s3_path.rstrip('/')}/"
            objects = wr.s3.list_objects(s3_path, boto3_session=cls.boto3_session)
            if objects:
                cls.log.info(f"Deleting S3 Objects at {s3_path}...")
                wr.s3.delete_objects(objects, boto3_session=cls.boto3_session)

        # Delete any dataframes that were stored in the Dataframe Cache
        cls.log.info("Deleting Dataframe Cache...")
        cls.df_cache.delete_recursive(endpoint_name)

        # Delete the endpoint
        time.sleep(2)  # Allow AWS to catch up
        try:
            cls.log.info(f"Deleting Endpoint {endpoint_name}...")
            cls.sm_client.delete_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            cls.log.error("Error deleting endpoint.")
            raise e

        time.sleep(5)  # Final sleep for AWS to fully register deletions

    @classmethod
    def delete_endpoint_models(cls, endpoint_name: str):
        """Delete the underlying Model for an Endpoint

        Args:
            endpoint_name (str): The name of the endpoint to delete
        """

        # Grab the Endpoint Config Name from AWS
        endpoint_config_name = cls.sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]

        # Retrieve the Model Names from the Endpoint Config
        try:
            endpoint_config = cls.sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        except botocore.exceptions.ClientError:
            cls.log.info(f"Endpoint Config {endpoint_config_name} doesn't exist...")
            return
        model_names = [variant["ModelName"] for variant in endpoint_config["ProductionVariants"]]
        for model_name in model_names:
            cls.log.info(f"Deleting Internal Model {model_name}...")
            try:
                cls.sm_client.delete_model(ModelName=model_name)
            except botocore.exceptions.ClientError as error:
                error_code = error.response["Error"]["Code"]
                error_message = error.response["Error"]["Message"]
                if error_code == "ResourceInUse":
                    cls.log.warning(f"Model {model_name} is still in use...")
                else:
                    cls.log.warning(f"Error: {error_code} - {error_message}")


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from workbench.api import FeatureSet
    from workbench.utils.endpoint_utils import fs_evaluation_data
    import random

    # Grab an EndpointCore object and pull some information from it
    my_endpoint = EndpointCore("abalone-regression")

    # Let's do a check/validation of the Endpoint
    assert my_endpoint.exists()

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Test onboarding
    my_endpoint.onboard()

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.get_tags()}")

    print("Details:")
    print(f"{my_endpoint.details()}")

    # Serverless?
    print(f"Serverless: {my_endpoint.is_serverless()}")

    # Health Check
    print(f"Health Check: {my_endpoint.health_check()}")

    # Get the ARN
    print(f"ARN: {my_endpoint.arn()}")

    # Get the internal model data URL
    print(f"Internal Model Data URL: {my_endpoint.model_data_url()}")

    # Capitalization Test
    fs = FeatureSet("abalone_features")
    df = fs.pull_dataframe()[:100]
    cap_df = df.copy()
    cap_df.columns = [col.upper() for col in cap_df.columns]
    my_endpoint._predict(cap_df)

    # Boolean Type Test
    df["bool_column"] = [random.choice([True, False]) for _ in range(len(df))]
    result_df = my_endpoint._predict(df)
    assert result_df["bool_column"].dtype == bool

    # Run Auto Inference on the Endpoint (uses the FeatureSet)
    print("Running Auto Inference...")
    my_endpoint.auto_inference()

    # Run Inference where we provide the data
    # Note: This dataframe could be from a FeatureSet or any other source
    print("Running Inference...")
    my_eval_df = fs_evaluation_data(my_endpoint)
    pred_results = my_endpoint.inference(my_eval_df)

    # Now set capture=True to save inference results and metrics
    my_eval_df = fs_evaluation_data(my_endpoint)
    pred_results = my_endpoint.inference(my_eval_df, capture_uuid="holdout_xyz")

    # Run Inference and metrics for a Classification Endpoint
    class_endpoint = EndpointCore("wine-classification")
    auto_predictions = class_endpoint.auto_inference()

    # Generate the confusion matrix
    target = "solubility_class"
    print(class_endpoint.generate_confusion_matrix(target, auto_predictions))

    # Run predictions using the fast_inference method
    fast_results = my_endpoint.fast_inference(my_eval_df)

    # Test the class method delete
    EndpointCore.managed_delete("abc")
