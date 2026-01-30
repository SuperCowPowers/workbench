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
from sklearn.metrics import confusion_matrix
from workbench.utils.metrics_utils import compute_regression_metrics, compute_classification_metrics

# SageMaker Imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts import FeatureSetCore, ModelCore, ModelType, ModelFramework
from workbench.utils.endpoint_metrics import EndpointMetrics
from workbench.utils.cache import Cache
from workbench.utils.s3_utils import compute_s3_object_hash
from workbench.utils.model_utils import uq_metrics
from workbench.utils.xgboost_model_utils import pull_cv_results as xgboost_pull_cv
from workbench.utils.pytorch_utils import pull_cv_results as pytorch_pull_cv
from workbench.utils.chemprop_utils import pull_cv_results as chemprop_pull_cv
from workbench_bridges.endpoints.fast_inference import fast_inference


class EndpointCore(Artifact):
    """EndpointCore: Workbench EndpointCore Class

    Common Usage:
        ```python
        my_endpoint = EndpointCore(endpoint_name)
        prediction_df = my_endpoint.predict(test_df)
        metrics = my_endpoint.regression_metrics(target_column, prediction_df)
        for metric, value in metrics.items():
            print(f"{metric}: {value:0.3f}")
        ```
    """

    def __init__(self, endpoint_name, **kwargs):
        """EndpointCore Initialization

        Args:
            endpoint_name (str): Name of Endpoint in Workbench
        """

        # Make sure the endpoint_name is a valid name
        self.is_name_valid(endpoint_name, delimiter="-", lower_case=False)

        # Call SuperClass Initialization
        super().__init__(endpoint_name, **kwargs)

        # Grab an Cloud Metadata object and pull information for Endpoints
        self.endpoint_name = endpoint_name
        self.endpoint_meta = self.meta.endpoint(self.endpoint_name)

        # Sanity check that we found the endpoint
        if self.endpoint_meta is None:
            self.log.important(f"Could not find endpoint {self.name} within current visibility scope")
            return

        # Sanity check the Endpoint state
        if self.endpoint_meta["EndpointStatus"] == "Failed":
            self.log.critical(f"Endpoint {self.name} is in a failed state")
            reason = self.endpoint_meta["FailureReason"]
            self.log.critical(f"Failure Reason: {reason}")
            self.log.critical("Please delete this endpoint and re-deploy...")

        # Set the Inference, Capture, and Monitoring S3 Paths
        base_endpoint_path = f"{self.endpoints_s3_path}/{self.name}"
        self.endpoint_inference_path = f"{base_endpoint_path}/inference"
        self.endpoint_data_capture_path = f"{base_endpoint_path}/data_capture"
        self.endpoint_monitoring_path = f"{base_endpoint_path}/monitoring"

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

    def data_capture(self):
        """Get the MonitorCore class for this endpoint"""
        from workbench.core.artifacts.data_capture_core import DataCaptureCore

        return DataCaptureCore(self.endpoint_name)

    def enable_data_capture(self):
        """Add data capture to the endpoint"""
        self.data_capture().enable()

    def monitor(self):
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
        metrics_key = f"endpoint:{self.name}:endpoint_metrics"
        endpoint_metrics = self.temp_storage.get(metrics_key)
        if endpoint_metrics is not None:
            return endpoint_metrics

        # We don't have it cached so let's get it from CloudWatch
        if "ProductionVariants" not in self.endpoint_meta:
            return None
        self.log.important("Updating endpoint metrics...")
        variant = self.endpoint_meta["ProductionVariants"][0]["VariantName"]
        endpoint_metrics = EndpointMetrics().get_metrics(self.name, variant=variant)
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

    def is_monitored(self) -> bool:
        """Is monitoring enabled for this Endpoint?

        Returns:
            True if monitoring is enabled, False otherwise.
        """
        try:
            response = self.sm_client.list_monitoring_schedules(EndpointName=self.name)
            return bool(response.get("MonitoringScheduleSummaries", []))
        except ClientError:
            return False

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

    def auto_inference(self) -> pd.DataFrame:
        """Run inference on the endpoint using the test data from the model training view"""

        # Sanity Check that we have a model
        model = ModelCore(self.get_input())
        if not model.exists():
            self.log.error("No model found for this endpoint. Returning empty DataFrame.")
            return pd.DataFrame()

        # Grab the evaluation data from the Model's training view
        all_df = model.training_view().pull_dataframe()
        eval_df = all_df[~all_df["training"]]

        # Remove AWS created columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        eval_df = eval_df.drop(columns=aws_cols, errors="ignore")

        # Run inference
        return self.inference(eval_df, "auto_inference")

    def full_inference(self) -> pd.DataFrame:
        """Run inference on the endpoint using all the data from the model training view"""

        # Sanity Check that we have a model
        model = ModelCore(self.get_input())
        if not model.exists():
            self.log.error("No model found for this endpoint. Returning empty DataFrame.")
            return pd.DataFrame()

        # Grab the full data from the Model's training view
        eval_df = model.training_view().pull_dataframe()

        # Remove AWS created columns
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        eval_df = eval_df.drop(columns=aws_cols, errors="ignore")

        # Run inference
        return self.inference(eval_df, "full_inference")

    def inference(
        self,
        eval_df: pd.DataFrame,
        capture_name: str = None,
        id_column: str = None,
        drop_error_rows: bool = False,
        include_quantiles: bool = False,
    ) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
            capture_name (str, optional): Name of the inference capture (default=None)
            id_column (str, optional): Name of the ID column (default=None)
            drop_error_rows (bool, optional): If True, drop rows that had endpoint errors/issues (default=False)
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: DataFrame with the inference results

        Note:
            If capture=True inference/performance metrics are written to S3 Endpoint Inference Folder
        """

        # Check if this is a 'floating endpoint' (no model)
        if self.get_input() == "unknown":
            self.log.important("No model associated with this endpoint, running 'no frills' inference...")
            return self.fast_inference(eval_df)

        # Grab the model features and target column
        model = ModelCore(self.model_name)
        features = model.features()
        targets = model.target()  # Note: We have multi-target models (so this could be a list)

        # Run predictions on the evaluation data
        prediction_df = self._predict(eval_df, features, drop_error_rows)
        if prediction_df.empty:
            self.log.warning("No predictions were made. Returning empty DataFrame.")
            return prediction_df

        # Normalize targets to handle both string and list formats
        if isinstance(targets, list):
            primary_target = targets[0] if targets else None
        else:
            primary_target = targets

        # Sanity Check that the target column is present
        if primary_target not in prediction_df.columns:
            self.log.important(f"Target Column {primary_target} not found in prediction_df!")
            self.log.important("In order to compute metrics, the target column must be present!")
            metrics = pd.DataFrame()

        # Compute the standard performance metrics for this model
        else:
            if model.model_type in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]:
                prediction_df = self.residuals(primary_target, prediction_df)
                metrics = self.regression_metrics(primary_target, prediction_df)
            elif model.model_type == ModelType.CLASSIFIER:
                metrics = self.classification_metrics(primary_target, prediction_df)
            else:
                # For other model types, we don't compute metrics
                self.log.info(f"Model Type: {model.model_type} doesn't have metrics...")
                metrics = pd.DataFrame()

        # Print out the metrics
        print(f"Performance Metrics for {self.model_name} on {self.name}")
        print(metrics.head())

        # Capture the inference results and metrics
        if primary_target and capture_name:

            # If we don't have an id_column, we'll pull it from the model's FeatureSet
            if id_column is None:
                fs = FeatureSetCore(model.get_input())
                id_column = fs.id_column

            # Normalize targets to a list for iteration
            target_list = targets if isinstance(targets, list) else [targets]
            primary_target = target_list[0]

            # For single-target models (99% of cases), just save with capture_name
            # For multi-target models, save each as {prefix}_{target} plus primary as capture_name
            is_multi_target = len(target_list) > 1

            if is_multi_target:
                prefix = "auto" if capture_name == "auto_inference" else capture_name

            for target in target_list:
                # Drop rows with NaN target values for metrics/plots
                target_df = prediction_df.dropna(subset=[target])

                # For multi-target models, prediction column is {target}_pred, otherwise "prediction"
                pred_col = f"{target}_pred" if is_multi_target else "prediction"

                # Compute per-target metrics
                if model.model_type in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]:
                    target_metrics = self.regression_metrics(target, target_df, prediction_col=pred_col)
                elif model.model_type == ModelType.CLASSIFIER:
                    target_metrics = self.classification_metrics(target, target_df, prediction_col=pred_col)
                else:
                    target_metrics = pd.DataFrame()

                if is_multi_target:
                    # Multi-target: save as {prefix}_{target}
                    target_capture_name = f"{prefix}_{target}"
                    description = target_capture_name.replace("_", " ").title()
                    self._capture_inference_results(
                        target_capture_name,
                        target_df,
                        target,
                        model.model_type,
                        target_metrics,
                        description,
                        features,
                        id_column,
                        include_quantiles,
                    )

                # Save primary target (or single target) with original capture_name
                if target == primary_target:
                    self._capture_inference_results(
                        capture_name,
                        target_df,
                        target,
                        model.model_type,
                        target_metrics,
                        capture_name.replace("_", " ").title(),
                        features,
                        id_column,
                        include_quantiles,
                    )

            # Capture uncertainty metrics if prediction_std is available (UQ, ChemProp, etc.)
            if "prediction_std" in prediction_df.columns:
                metrics = uq_metrics(prediction_df, primary_target)
                self.param_store.upsert(f"/workbench/models/{model.name}/inference/{capture_name}", metrics)

        # Return the prediction DataFrame
        return prediction_df

    def cross_fold_inference(self, include_quantiles: bool = False) -> pd.DataFrame:
        """Pull cross-fold inference training results for this Endpoint's model

        Args:
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: A DataFrame with cross fold predictions
        """

        # Grab our model
        model = ModelCore(self.model_name)

        # Compute CrossFold (Metrics and Prediction Dataframe)
        # For PyTorch and ChemProp, pull pre-computed CV results from training
        if model.model_framework in [ModelFramework.UNKNOWN, ModelFramework.XGBOOST]:
            cross_fold_metrics, out_of_fold_df = xgboost_pull_cv(model)
        elif model.model_framework == ModelFramework.PYTORCH:
            cross_fold_metrics, out_of_fold_df = pytorch_pull_cv(model)
        elif model.model_framework == ModelFramework.CHEMPROP:
            cross_fold_metrics, out_of_fold_df = chemprop_pull_cv(model)
        else:
            self.log.error(f"Cross-Fold Inference not supported for Model Framework: {model.model_framework}.")
            return pd.DataFrame()

        # If the metrics dataframe isn't empty save to the param store
        if not cross_fold_metrics.empty:
            # Convert to list of dictionaries
            metrics = cross_fold_metrics.to_dict(orient="records")
            self.param_store.upsert(f"/workbench/models/{model.name}/inference/cross_fold", metrics)

        # If the out_of_fold_df is empty return it
        if out_of_fold_df.empty:
            self.log.warning("No out-of-fold predictions were made. Returning empty DataFrame.")
            return out_of_fold_df

        # Capture the results
        targets = model.target()  # Note: We have multi-target models (so this could be a list)
        model_type = model.model_type

        # Get the id_column from the model's FeatureSet
        fs = FeatureSetCore(model.get_input())
        id_column = fs.id_column

        # Normalize targets to a list for iteration
        target_list = targets if isinstance(targets, list) else [targets]
        primary_target = target_list[0]

        # If we don't have a smiles column, try to merge it from the FeatureSet
        if "smiles" not in out_of_fold_df.columns:
            fs_df = fs.query(f'SELECT {fs.id_column}, "smiles" FROM "{fs.athena_table}"')
            if "smiles" in fs_df.columns:
                self.log.info("Merging 'smiles' column from FeatureSet into out-of-fold predictions.")
                out_of_fold_df = out_of_fold_df.merge(fs_df, on=fs.id_column, how="left")

        # Collect UQ columns (q_*, confidence) for additional tracking (used for hashing)
        additional_columns = [col for col in out_of_fold_df.columns if col.startswith("q_") or col == "confidence"]
        if additional_columns:
            self.log.info(f"UQ columns from training: {', '.join(additional_columns)}")

        # Capture uncertainty metrics if prediction_std is available (UQ, ChemProp, etc.)
        if "prediction_std" in out_of_fold_df.columns:
            metrics = uq_metrics(out_of_fold_df, primary_target)
            self.param_store.upsert(f"/workbench/models/{model.name}/inference/full_cross_fold", metrics)

        # For single-target models (99% of cases), just save as "full_cross_fold"
        # For multi-target models, save each as cv_{target} plus primary as "full_cross_fold"
        is_multi_target = len(target_list) > 1
        for target in target_list:
            # Drop rows with NaN target values for metrics/plots
            target_df = out_of_fold_df.dropna(subset=[target])

            # For multi-target models, prediction column is {target}_pred, otherwise "prediction"
            pred_col = f"{target}_pred" if is_multi_target else "prediction"

            # Compute per-target metrics
            if model_type in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]:
                target_metrics = self.regression_metrics(target, target_df, prediction_col=pred_col)
            elif model_type == ModelType.CLASSIFIER:
                target_metrics = self.classification_metrics(target, target_df, prediction_col=pred_col)
            else:
                target_metrics = pd.DataFrame()

            if is_multi_target:
                # Multi-target: save as cv_{target}
                capture_name = f"cv_{target}"
                description = capture_name.replace("_", " ").title()
                self._capture_inference_results(
                    capture_name,
                    target_df,
                    target,
                    model_type,
                    target_metrics,
                    description,
                    features=additional_columns,
                    id_column=id_column,
                    include_quantiles=include_quantiles,
                )

            # Save primary target (or single target) as "full_cross_fold"
            if target == primary_target:
                self._capture_inference_results(
                    "full_cross_fold",
                    target_df,
                    target,
                    model_type,
                    target_metrics,
                    "Full Cross Fold",
                    features=additional_columns,
                    id_column=id_column,
                    include_quantiles=include_quantiles,
                )

        return out_of_fold_df

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
        return fast_inference(self.name, eval_df, self.sm_session, threads=threads)

    def _predict(self, eval_df: pd.DataFrame, features: list[str], drop_error_rows: bool = False) -> pd.DataFrame:
        """Internal: Run prediction on observations in the given DataFrame

        Args:
            eval_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
            features (list[str]): List of feature column names needed for prediction
            drop_error_rows (bool): If True, drop rows that had endpoint errors/issues (default=False)
        Returns:
            pd.DataFrame: Return the DataFrame with additional columns, prediction and any _proba columns
        """

        # Sanity check: Does the DataFrame have 0 rows?
        if eval_df.empty:
            self.log.warning("Evaluation DataFrame has 0 rows. No predictions to run.")
            return pd.DataFrame(columns=eval_df.columns)  # Return empty DataFrame with same structure

        # Sanity check: Does the DataFrame have the required features?
        df_columns_lower = set(col.lower() for col in eval_df.columns)
        features_lower = set(feature.lower() for feature in features)
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
            partial_df = self._endpoint_error_handling(predictor, eval_df[index : index + 100], drop_error_rows)
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

    def _endpoint_error_handling(self, predictor, feature_df, drop_error_rows: bool = False) -> pd.DataFrame:
        """Internal: Handles errors, retries, and binary search for problematic rows.

        Args:
            predictor (Predictor): The SageMaker Predictor object
            feature_df (pd.DataFrame): DataFrame to run predictions on
            drop_error_rows (bool): If True, drop rows that had endpoint errors/issues (default=False)
        Returns:
            pd.DataFrame: DataFrame with predictions (NaNs for problematic rows or dropped rows if specified)
        """

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
                self.log.error(f"Error {error_code}")
                self.log.error(err.response)
                self.log.error("Model not ready. Sleeping and retrying...")
                time.sleep(60)
                return self._endpoint_error_handling(predictor, feature_df)

            elif error_code == "ModelError":
                # Log full error response to capture all available debugging info
                self.log.error(f"Error {error_code}")
                self.log.error(err.response)
                self.log.warning("Bisecting the DataFrame and retrying...")

                # Base case: single row handling
                if len(feature_df) == 1:
                    if not self.endpoint_return_columns:
                        raise
                    self.log.warning(f"Endpoint Inference failed on: {feature_df}")
                    if drop_error_rows:
                        self.log.warning("Dropping rows with endpoint errors...")
                        return pd.DataFrame(columns=feature_df.columns)
                    # Fill with NaNs for inference columns, keeping original feature data
                    self.log.warning("Filling with NaNs for inference columns...")
                    return self._fill_with_nans(feature_df)

                # Binary search for problematic rows
                mid_point = len(feature_df) // 2
                self.log.info(f"Bisect DataFrame: 0 -> {mid_point} and {mid_point} -> {len(feature_df)}")
                first_half = self._endpoint_error_handling(predictor, feature_df.iloc[:mid_point], drop_error_rows)
                second_half = self._endpoint_error_handling(predictor, feature_df.iloc[mid_point:], drop_error_rows)
                return pd.concat([first_half, second_half], ignore_index=True)

            else:
                self.log.critical(f"Unexpected ClientError: {error_code}")
                self.log.critical(err.response)
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
            for column in set(feature_df.columns).intersection(self.endpoint_return_columns):
                # Use .iloc[0] to access the first row by position, regardless of the index
                one_row_df_with_nans.at[0, column] = feature_df.iloc[0][column]

        return one_row_df_with_nans

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame, hash_length: int = 8):
        # Internal: Compute a data hash for the dataframe
        if df.empty:
            return "--hash--"

        # Sort the dataframe by columns to ensure consistent ordering
        df = df.copy()
        df = df.sort_values(by=sorted(df.columns.tolist()))
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        combined = row_hashes.values.tobytes()
        return hashlib.md5(combined).hexdigest()[:hash_length]

    def _capture_inference_results(
        self,
        capture_name: str,
        pred_results_df: pd.DataFrame,
        target: str,
        model_type: ModelType,
        metrics: pd.DataFrame,
        description: str,
        features: list,
        id_column: str = None,
        include_quantiles: bool = False,
    ):
        """Internal: Capture the inference results and metrics to S3 for a single target

        Args:
            capture_name (str): Name of the inference capture
            pred_results_df (pd.DataFrame): DataFrame with the prediction results
            target (str): Target column name
            model_type (ModelType): Type of the model (e.g. REGRESSOR, CLASSIFIER)
            metrics (pd.DataFrame): DataFrame with the performance metrics
            description (str): Description of the inference results
            features (list): List of features to include in the inference results
            id_column (str, optional): Name of the ID column (default=None)
            include_quantiles (bool): Include q_* quantile columns in output (default: False)
        """

        # Compute a dataframe hash (just use the last 8)
        data_hash = self._hash_dataframe(pred_results_df[features])

        # Metadata for the model inference
        inference_meta = {
            "name": capture_name,
            "data_hash": data_hash,
            "num_rows": len(pred_results_df),
            "description": description,
        }

        # Create the S3 Path for the Inference Capture
        inference_capture_path = f"{self.endpoint_inference_path}/{capture_name}"

        # Write the metadata dictionary and metrics to our S3 Model Inference Folder
        wr.s3.to_json(
            pd.DataFrame([inference_meta]),
            f"{inference_capture_path}/inference_meta.json",
            index=False,
        )
        self.log.info(f"Writing metrics to {inference_capture_path}/inference_metrics.csv")
        wr.s3.to_csv(metrics, f"{inference_capture_path}/inference_metrics.csv", index=False)

        # Save the inference predictions for this target
        self._save_target_inference(inference_capture_path, pred_results_df, target, id_column, include_quantiles)

        # CLASSIFIER: Write the confusion matrix to our S3 Model Inference Folder
        if model_type == ModelType.CLASSIFIER:
            conf_mtx = self.generate_confusion_matrix(target, pred_results_df)
            self.log.info(f"Writing confusion matrix to {inference_capture_path}/inference_cm.csv")
            # Note: Unlike other dataframes here, we want to write the index (labels) to the CSV
            wr.s3.to_csv(conf_mtx, f"{inference_capture_path}/inference_cm.csv", index=True)

        # Now recompute the details for our Model
        self.log.important(f"Loading inference metrics for {self.model_name}...")
        model = ModelCore(self.model_name)
        model._load_inference_metrics(capture_name)

    def _save_target_inference(
        self,
        inference_capture_path: str,
        pred_results_df: pd.DataFrame,
        target: str,
        id_column: str = None,
        include_quantiles: bool = False,
    ):
        """Save inference results for a single target.

        Args:
            inference_capture_path (str): S3 path for inference capture
            pred_results_df (pd.DataFrame): DataFrame with prediction results
            target (str): Target column name
            id_column (str, optional): Name of the ID column
            include_quantiles (bool): Include q_* quantile columns in output (default: False)
        """
        cols = pred_results_df.columns

        # Build output columns: id, target, prediction, prediction_std, UQ columns, proba columns
        output_columns = []
        if id_column and id_column in cols:
            output_columns.append(id_column)
        if target and target in cols:
            output_columns.append(target)

        output_columns += [c for c in ["prediction", "prediction_std"] if c in cols]

        # Add confidence column (always include if present)
        if "confidence" in cols:
            output_columns.append("confidence")

        # Add quantile columns (q_*) only if requested
        if include_quantiles:
            output_columns += [c for c in cols if c.startswith("q_")]

        # Add proba columns for classifiers
        output_columns += [c for c in cols if c.endswith("_proba")]

        # Add smiles column if present
        if "smiles" in cols:
            output_columns.append("smiles")

        # Write the predictions to S3
        output_file = f"{inference_capture_path}/inference_predictions.csv"
        self.log.info(f"Writing predictions to {output_file}")
        wr.s3.to_csv(pred_results_df[output_columns], output_file, index=False)

    def regression_metrics(
        self, target_column: str, prediction_df: pd.DataFrame, prediction_col: str = "prediction"
    ) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint

        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
            prediction_col (str): Name of the prediction column (default: "prediction")

        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """
        return compute_regression_metrics(prediction_df, target_column, prediction_col)

    def residuals(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Add the residuals to the prediction DataFrame
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with two new columns called 'residuals' and 'residuals_abs'
        """
        # Check for prediction column
        if "prediction" not in prediction_df.columns:
            self.log.warning("No 'prediction' column found. Cannot compute residuals.")
            return prediction_df

        y_true = prediction_df[target_column]
        y_pred = prediction_df["prediction"]

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

    def classification_metrics(
        self, target_column: str, prediction_df: pd.DataFrame, prediction_col: str = "prediction"
    ) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint

        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
            prediction_col (str): Name of the prediction column (default: "prediction")

        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """
        # Get class labels from the model (metrics_utils will infer if None)
        class_labels = ModelCore(self.model_name).class_labels()
        return compute_classification_metrics(prediction_df, target_column, class_labels, prediction_col)

    def generate_confusion_matrix(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the confusion matrix for this Endpoint

        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results

        Returns:
            pd.DataFrame: DataFrame with the confusion matrix
        """
        # Check for prediction column
        if "prediction" not in prediction_df.columns:
            self.log.warning("No 'prediction' column found in DataFrame")
            return pd.DataFrame()

        # Drop rows with NaN predictions (can't include in confusion matrix)
        nan_mask = prediction_df["prediction"].isna()
        if nan_mask.any():
            n_nan = nan_mask.sum()
            self.log.warning(f"Dropping {n_nan} rows with NaN predictions for confusion matrix")
            prediction_df = prediction_df[~nan_mask].copy()

        y_true = prediction_df[target_column]
        y_pred = prediction_df["prediction"]

        # Get model class labels
        model_class_labels = ModelCore(self.model_name).class_labels()

        # Use model labels if available, otherwise infer from data
        if model_class_labels:
            self.log.important("Using model class labels for confusion matrix ordering...")
            labels = model_class_labels
        else:
            labels = sorted(list(set(y_true) | set(y_pred)))

        # Compute confusion matrix and create DataFrame
        conf_mtx = confusion_matrix(y_true, y_pred, labels=labels)
        conf_mtx_df = pd.DataFrame(conf_mtx, index=labels, columns=labels)
        conf_mtx_df.index.name = "labels"
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
            self.log.warning(f"Endpoint {self.name}: Does not allow manual override of the input!")
            return

        # Okay we're going to allow this to be set
        self.log.important(f"{self.name}: Setting input to {input}...")
        self.log.important("Be careful with this! It breaks automatic provenance of the artifact!")
        self.upsert_workbench_meta({"workbench_input": input})

    def delete(self):
        """Delete an existing Endpoint: Underlying Models, Configuration, and Endpoint"""
        if not self.exists():
            self.log.warning(f"Trying to delete an Endpoint that doesn't exist: {self.name}")

        # Remove this endpoint from the list of registered endpoints
        self.log.info(f"Removing {self.name} from the list of registered endpoints...")
        ModelCore(self.model_name).remove_endpoint(self.name)

        # Call the Class Method to delete the Endpoint
        EndpointCore.managed_delete(endpoint_name=self.name)

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

        # Recursively delete all endpoint S3 artifacts (inference, etc)
        # Note: We do not want to delete the data_capture/ files since these
        #       might be used for collection and data drift analysis
        base_endpoint_path = f"{cls.endpoints_s3_path}/{endpoint_name}/"
        all_s3_objects = wr.s3.list_objects(base_endpoint_path, boto3_session=cls.boto3_session)

        # Filter out objects that contain 'data_capture/' in their path
        s3_objects_to_delete = [obj for obj in all_s3_objects if "/data_capture/" not in obj]
        cls.log.info(f"Found {len(all_s3_objects)} total objects at {base_endpoint_path}")
        cls.log.info(f"Filtering out data_capture files, will delete {len(s3_objects_to_delete)} objects...")
        cls.log.info(f"Objects to delete: {s3_objects_to_delete}")

        if s3_objects_to_delete:
            wr.s3.delete_objects(s3_objects_to_delete, boto3_session=cls.boto3_session)
            cls.log.info(f"Successfully deleted {len(s3_objects_to_delete)} objects")
        else:
            cls.log.info("No objects to delete (only data_capture files found)")

        # Delete any dataframes that were stored in the Dataframe Cache
        cls.log.info("Deleting Dataframe Cache...")
        cls.df_cache.delete_recursive(endpoint_name)

        # Delete the endpoint
        cls.log.info(f"Deleting Endpoint {endpoint_name}...")
        time.sleep(10)  # Allow AWS to catch up
        try:
            cls.sm_client.delete_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            cls.log.error("Error deleting endpoint.")
            raise e

        time.sleep(10)  # Final sleep for AWS to fully register deletions

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
    from workbench.utils.endpoint_utils import get_evaluation_data
    import random

    # Grab an EndpointCore object and pull some information from it
    my_endpoint = EndpointCore("abalone-regression")

    # Test various error conditions (set row 42 length to pd.NA)
    # Note: This test should return ALL rows
    my_eval_df = get_evaluation_data(my_endpoint)
    my_eval_df.at[42, "length"] = pd.NA
    pred_results = my_endpoint.inference(my_eval_df, drop_error_rows=True)
    print(f"Sent rows: {len(my_eval_df)}")
    print(f"Received rows: {len(pred_results)}")
    assert len(pred_results) == len(my_eval_df), "Predictions should match the number of sent rows"

    # Now we put in an invalid value
    print("*" * 80)
    print("NOW TESTING ERROR CONDITIONS...")
    print("*" * 80)
    my_eval_df.at[42, "length"] = "invalid_value"
    pred_results = my_endpoint.inference(my_eval_df, drop_error_rows=True)
    print(f"Sent rows: {len(my_eval_df)}")
    print(f"Received rows: {len(pred_results)}")
    assert len(pred_results) < len(my_eval_df), "Predictions should be less than the number of sent rows"

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
    my_endpoint.inference(cap_df)

    # Boolean Type Test
    df["bool_column"] = [random.choice([True, False]) for _ in range(len(df))]
    result_df = my_endpoint.inference(df)
    assert result_df["bool_column"].dtype == bool

    # Missing Feature Test
    missing_df = df.drop(columns=["length"])
    try:
        my_endpoint.inference(missing_df)
    except ValueError as e:
        print(f"Expected error for missing feature: {e}")

    # Run Auto Inference on the Endpoint (uses the FeatureSet)
    print("Running Auto Inference...")
    my_endpoint.auto_inference()

    # Run Inference where we provide the data
    # Note: This dataframe could be from a FeatureSet or any other source
    print("Running Inference...")
    my_eval_df = get_evaluation_data(my_endpoint)
    pred_results = my_endpoint.inference(my_eval_df)

    # Now set capture=True to save inference results and metrics
    my_eval_df = get_evaluation_data(my_endpoint)
    pred_results = my_endpoint.inference(my_eval_df, capture_name="holdout_xyz")

    # Run predictions using the fast_inference method
    fast_results = my_endpoint.fast_inference(my_eval_df)

    # Test the cross_fold_inference method
    print("Running Cross-Fold Inference...")
    all_results = my_endpoint.cross_fold_inference()
    print(all_results)

    # Run Inference and metrics for a Classification Endpoint
    class_endpoint = EndpointCore("wine-classification")
    auto_predictions = class_endpoint.auto_inference()

    # Generate the confusion matrix
    target = "wine_class"
    print(class_endpoint.generate_confusion_matrix(target, auto_predictions))

    # Test the cross_fold_inference method
    print("Running Cross-Fold Inference...")
    all_results = class_endpoint.cross_fold_inference()
    print(all_results)
    print("All done...")

    # Test the class method delete (commented out for now)
    # from workbench.api import Model
    # model = Model("abalone-regression")
    # model.to_endpoint("test-endpoint")
    # EndpointCore.managed_delete("test-endpoint")
