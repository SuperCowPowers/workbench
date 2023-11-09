"""Endpoint: SageWorks Endpoint Class"""
import sys
from datetime import datetime
import botocore
import pandas as pd
import numpy as np
from io import StringIO
import awswrangler as wr
import ast

# Model Performance Scores
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    precision_recall_fscore_support,
    median_absolute_error,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer
from math import sqrt

from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.artifacts.models.model import Model, ModelType
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.endpoint_metrics import EndpointMetrics


class Endpoint(Artifact):
    """Endpoint: SageWorks Endpoint Class

    Common Usage:
        my_endpoint = Endpoint(endpoint_uuid)
        prediction_df = my_endpoint.predict(test_df)
        metrics = my_endpoint.regression_metrics(target_column, prediction_df)
        for metric, value in metrics.items():
            print(f"{metric}: {value:0.3f}")
    """

    def __init__(self, endpoint_uuid, force_refresh: bool = False, exit_on_error=True):
        """Endpoint Initialization

        Args:
            endpoint_uuid (str): Name of Endpoint in SageWorks
            force_refresh (bool, optional): Force a refresh of the AWS Broker. Defaults to False.
        """
        # Call SuperClass Initialization
        super().__init__(endpoint_uuid)

        # Grab an AWS Metadata Broker object and pull information for Endpoints
        self.endpoint_name = endpoint_uuid
        self.endpoint_meta = self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS, force_refresh=force_refresh).get(
            self.endpoint_name
        )

        # Sanity check and then set up our FeatureSet attributes
        if self.endpoint_meta is None:
            self.log.important(f"Could not find endpoint {self.uuid} within current visibility scope")
            return

        self.endpoint_return_columns = None
        self.exit_on_error = exit_on_error

        # Set the Model Training and Inference S3 Paths
        self.model_name = self.get_input()
        self.model_training_path = self.models_s3_path + "/training"
        self.model_inference_path = self.models_s3_path + "/inference"

        # All done
        self.log.info(f"Endpoint Initialized: {self.endpoint_name}")

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self.endpoint_meta = self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS, force_refresh=True).get(
            self.endpoint_name
        )

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.endpoint_meta is None:
            self.log.info(f"Endpoint {self.endpoint_name} not found in AWS Metadata!")
            return False
        return True

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference/prediction on the given Feature DataFrame
        Args:
            feature_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
        Returns:
            pd.DataFrame: Return the feature DataFrame with two additional columns (prediction, pred_proba)
        """

        # Create our Endpoint Predictor Class
        predictor = Predictor(
            self.endpoint_name,
            sagemaker_session=self.sm_session,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
        )

        # Now split up the dataframe into 500 row chunks, send those chunks to our
        # endpoint (with error handling) and stitch all the chunks back together
        df_list = []
        for index in range(0, len(feature_df), 500):
            print("Processing...")

            # Compute partial DataFrames, add them to a list, and concatenate at the end
            partial_df = self._endpoint_error_handling(predictor, feature_df[index : index + 500])
            df_list.append(partial_df)

        # Concatenate the dataframes
        combined_df = pd.concat(df_list, ignore_index=True)

        # Convert data to numeric
        # Note: Since we're using CSV serializers numeric columns often get changed to generic 'object' types

        # Hard Conversion
        # Note: If are string/object columns we want to use 'ignore' here so those columns
        #       won't raise an error (columns maintain current type)
        converted_df = combined_df.apply(pd.to_numeric, errors="ignore")

        # Soft Conversion
        # Convert columns to the best possible dtype that supports the pd.NA missing value.
        converted_df = converted_df.convert_dtypes()

        # Return the Dataframe
        return converted_df

    def _endpoint_error_handling(self, predictor, feature_df):
        """Internal: Method that handles Errors, Retries, and Binary Search for Error Row(s)"""

        # Convert the DataFrame into a CSV buffer
        csv_buffer = StringIO()
        feature_df.to_csv(csv_buffer, index=False)

        # Error Handling if the Endpoint gives back an error
        try:
            # Send the CSV Buffer to the predictor
            results = predictor.predict(csv_buffer.getvalue())

            # Construct a DataFrame from the results
            results_df = pd.DataFrame.from_records(results[1:], columns=results[0])

            # Capture the return columns
            self.endpoint_return_columns = results_df.columns.tolist()

            # Return the results dataframe
            return results_df

        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "ModelError":  # Model Error
                # Report the error
                self.log.critical(f"Endpoint prediction error: {err.response.get('Message')}")
                if self.exit_on_error:
                    sys.exit(1)

                # Base case: DataFrame with 1 Row
                if len(feature_df) == 1:
                    # If we don't have ANY known good results we're kinda screwed
                    if not self.endpoint_return_columns:
                        raise err

                    # Construct an Error DataFrame (one row of NaNs in the return columns)
                    results_df = self._error_df(feature_df, self.endpoint_return_columns)
                    return results_df

                # Recurse on binary splits of the dataframe
                num_rows = len(feature_df)
                split = int(num_rows / 2)
                first_half = self._endpoint_error_handling(predictor, feature_df[0:split])
                second_half = self._endpoint_error_handling(predictor, feature_df[split:num_rows])
                return pd.concat([first_half, second_half], ignore_index=True)

            else:
                print("Unknown Error from Prediction Endpoint")
                raise err

    def _error_df(self, df, all_columns):
        """Internal: Method to construct an Error DataFrame (a Pandas DataFrame with one row of NaNs)"""
        # Create a new dataframe with all NaNs
        error_df = pd.DataFrame(dict(zip(all_columns, [[np.NaN]] * len(self.endpoint_return_columns))))
        # Now set the original values for the incoming dataframe
        for column in df.columns:
            error_df[column] = df[column].values
        return error_df

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

    def details(self, recompute: bool = False) -> dict:
        """Additional Details about this Endpoint
        Args:
            recompute(bool): Recompute the details (default: False)
        Returns:
            dict(dict): A dictionary of details about this Endpoint
        """
        # Check if we have cached version of the FeatureSet Details
        details_key = f"endpoint:{self.uuid}:details"
        metrics_key = f"endpoint:{self.uuid}:endpoint_metrics"
        cached_details = self.data_storage.get(details_key)
        if cached_details and not recompute:
            # Return the cached details but first check if we need to update the endpoint metrics
            endpoint_metrics = self.temp_storage.get(metrics_key)
            if endpoint_metrics is None:
                self.log.important("Updating endpoint metrics...")
                variant = self.endpoint_meta["ProductionVariants"][0]["VariantName"]
                endpoint_metrics = EndpointMetrics().get_metrics(self.uuid, variant=variant)
                cached_details["endpoint_metrics"] = endpoint_metrics
                self.temp_storage.set(metrics_key, endpoint_metrics)
            else:
                cached_details["endpoint_metrics"] = endpoint_metrics
            return cached_details

        # Fill in all the details about this Endpoint
        details = self.summary()

        # Get details from our AWS Metadata
        details["status"] = self.endpoint_meta["EndpointStatus"]
        details["instance"] = self.endpoint_meta["InstanceType"]
        try:
            details["instance_count"] = self.endpoint_meta["ProductionVariants"][0]["CurrentInstanceCount"] or "-"
        except KeyError:
            details["instance_count"] = "-"
        details["variant"] = self.endpoint_meta["ProductionVariants"][0]["VariantName"]

        # Add the underlying model details
        details["model_name"] = self.model_name
        model_details = self.model_details()
        details["model_type"] = model_details.get("model_type", "unknown")
        details["model_metrics"] = model_details.get("model_metrics")
        details["confusion_matrix"] = model_details.get("confusion_matrix")
        details["regression_predictions"] = model_details.get("regression_predictions")
        details["inference_meta"] = model_details.get("inference_meta")

        # Add endpoint metrics from CloudWatch
        details["endpoint_metrics"] = EndpointMetrics().get_metrics(self.uuid, variant=details["variant"])
        self.temp_storage.set(metrics_key, details["endpoint_metrics"])

        # Cache the details
        self.data_storage.set(details_key, details)

        # Return the details
        return details

    def make_ready(self) -> bool:
        """This is a BLOCKING method that will wait until the Endpoint is ready"""
        self.details(recompute=True)
        self.set_status("ready")
        self.refresh_meta()
        return True

    def model_details(self) -> dict:
        """Return the details about the model used in this Endpoint"""
        if self.model_name == "unknown":
            return {}
        else:
            model = Model(self.model_name)
            if model.exists():
                return model.details()
            else:
                return {}

    def model_type(self) -> str:
        """Return the type of model used in this Endpoint"""
        return self.details().get("model_type", "unknown")

    def capture_performance_metrics(
        self, feature_df: pd.DataFrame, target_column: str, data_name: str, data_hash: str, description: str
    ) -> None:
        """Capture the performance metrics for this Endpoint
        Args:
            feature_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
            target_column (str): Name of the target column
            data_name (str): Name of the data used for inference
            data_hash (str): Hash of the data used for inference
            description (str): Description of the data used for inference
        Returns:
            None
        Note:
            This method captures performance metrics and writes them to the S3 Model Inference Folder
        """

        # Run predictions on the feature_df
        prediction_df = self.predict(feature_df)

        # Compute the metrics
        model_type = self.model_type()
        if model_type == ModelType.REGRESSOR.value:
            metrics = self.regression_metrics(target_column, prediction_df)
        elif model_type == ModelType.CLASSIFIER.value:
            metrics = self.classification_metrics(target_column, prediction_df)
        else:
            raise ValueError(f"Unknown Model Type: {model_type}")

        # Metadata for the model inference
        inference_meta = {
            "test_data": data_name,
            "test_data_hash": data_hash,
            "test_rows": len(feature_df),
            "description": description,
        }

        # Write the metadata dictionary, and metrics to our S3 Model Inference Folder
        wr.s3.to_json(
            pd.DataFrame([inference_meta]),
            f"{self.model_inference_path}/{self.model_name}/inference_meta.json",
            index=False,
        )
        wr.s3.to_csv(metrics, f"{self.model_inference_path}/{self.model_name}/inference_metrics.csv", index=False)

        # Write the confusion matrix to our S3 Model Inference Folder
        if model_type == ModelType.CLASSIFIER.value:
            conf_mtx = self.confusion_matrix(target_column, prediction_df)
            # Note: Unlike other dataframes here, we want to write the index (labels) to the CSV
            wr.s3.to_csv(conf_mtx, f"{self.model_inference_path}/{self.model_name}/inference_cm.csv", index=True)

        # Write the regression predictions to our S3 Model Inference Folder
        if model_type == ModelType.REGRESSOR.value:
            pred_df = self.regression_predictions(target_column, prediction_df)
            wr.s3.to_csv(
                pred_df, f"{self.model_inference_path}/{self.model_name}/inference_predictions.csv", index=False
            )

        # Recompute the details so that inference model metrics are updated
        self.log.important(f"Recomputing Details for {self.uuid} to show latest Inference Results...")
        self.details(recompute=True)

        # Now recompute the details for our Model
        self.log.important(f"Recomputing Details for {self.model_name} to show latest Inference Results...")
        model = Model(self.model_name)
        model.details(recompute=True)

    @staticmethod
    def regression_metrics(target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """

        # Compute the metrics
        y_true = prediction_df[target_column]
        y_pred = prediction_df["prediction"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        # Mean Absolute Percentage Error
        mape = np.mean(np.where(y_true != 0, np.abs((y_true - y_pred) / y_true), np.abs(y_true - y_pred))) * 100
        # Median Absolute Error
        medae = median_absolute_error(y_true, y_pred)

        # Return the metrics
        return pd.DataFrame.from_records([{"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "MedAE": medae}])

    @staticmethod
    def classification_metrics(target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """

        # Get a list of unique labels
        labels = prediction_df[target_column].unique()

        # Calculate scores
        scores = precision_recall_fscore_support(
            prediction_df[target_column], prediction_df["prediction"], average=None, labels=labels
        )

        # Calculate ROC AUC
        # ROC-AUC score measures the model's ability to distinguish between classes;
        # - A value of 0.5 indicates no discrimination (equivalent to random guessing)
        # - A score close to 1 indicates high discriminative power

        # Convert 'pred_proba' column to a 2D NumPy array
        y_score = np.array([ast.literal_eval(x) for x in prediction_df["pred_proba"]], dtype=float)
        # y_score = np.array(prediction_df['pred_proba'].tolist())

        # One-hot encode the true labels
        lb = LabelBinarizer()
        lb.fit(prediction_df[target_column])
        y_true = lb.transform(prediction_df[target_column])

        roc_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average=None)

        # Put the scores into a dataframe
        score_df = pd.DataFrame(
            {
                target_column: labels,
                "precision": scores[0],
                "recall": scores[1],
                "fscore": scores[2],
                "roc_auc": roc_auc,
                "support": scores[3],
            }
        )

        # Sort the target labels
        score_df = score_df.sort_values(by=[target_column], ascending=True)
        return score_df

    def confusion_matrix(self, target_column: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the confusion matrix for this Endpoint
        Args:
            target_column (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the confusion matrix
        """

        y_true = prediction_df[target_column]
        y_pred = prediction_df["prediction"]

        # Compute the confusion matrix
        conf_mtx = confusion_matrix(y_true, y_pred)

        # Get unique labels
        labels = sorted(list(set(y_true) | set(y_pred)))

        # Create a DataFrame
        conf_mtx_df = pd.DataFrame(conf_mtx, index=labels, columns=labels)
        return conf_mtx_df

    def regression_predictions(self, target: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the regression predictions for this Endpoint
        Args:
            target (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the regression predictions
        """

        # Return the predictions
        return prediction_df[[target, "prediction"]]

    def delete(self):
        """Delete an existing Endpoint: Underlying Models, Configuration, and Endpoint"""
        self.delete_endpoint_models()
        try:
            self.log.info(f"Deleting Endpoint Config {self.uuid}...")
            self.sm_client.delete_endpoint_config(EndpointConfigName=self.uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint Config {self.uuid} doesn't exist...")
        try:
            self.log.info(f"Deleting Endpoint {self.uuid}...")
            self.sm_client.delete_endpoint(EndpointName=self.uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint {self.uuid} doesn't exist...")

        # Now delete any data in the Cache
        for key in self.data_storage.list_subkeys(f"endpoint:{self.uuid}"):
            self.log.info(f"Deleting Cache Key: {key}")
            self.data_storage.delete(key)

    def delete_endpoint_models(self):
        """Delete the underlying Model for an Endpoint"""

        # Retrieve the Model Names from the Endpoint Config
        try:
            endpoint_config = self.sm_client.describe_endpoint_config(EndpointConfigName=self.uuid)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint Config {self.uuid} doesn't exist...")
            return
        model_names = [variant["ModelName"] for variant in endpoint_config["ProductionVariants"]]
        for model_name in model_names:
            self.log.info(f"Deleting Model {model_name}...")
            self.sm_client.delete_model(ModelName=model_name)


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from sageworks.transforms.pandas_transforms.features_to_pandas import (
        FeaturesToPandas,
    )

    # Grab an Endpoint object and pull some information from it
    my_endpoint = Endpoint("abalone-regression-end")

    # Let's do a check/validation of the Endpoint
    assert my_endpoint.exists()

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.sageworks_tags()}")

    print("Details:")
    print(f"{my_endpoint.details(recompute=True)}")

    #
    # This section is all about INFERENCE TESTING
    INFERENCE_TESTING = False
    if INFERENCE_TESTING:
        REGRESSION = False
        if REGRESSION:
            my_endpoint = Endpoint("abalone-regression-end")
            feature_to_pandas = FeaturesToPandas("abalone_feature_set")
            my_target_column = "class_number_of_rings"
            data_name = ("abalone_holdout_2023_10_19",)
            data_hash = ("12345",)
            description = "Test Abalone Data"
        else:
            my_endpoint = Endpoint("wine-classification-end")
            feature_to_pandas = FeaturesToPandas("wine_features")
            my_target_column = "wine_class"
            data_name = ("wine_holdout_2023_10_19",)
            data_hash = ("67890",)
            description = "Test Wine Data"

        # Transform the DataSource into a Pandas DataFrame (with max_rows = 500)
        feature_to_pandas.transform(max_rows=500)

        # Grab the output and show it
        my_feature_df = feature_to_pandas.get_output()
        print(my_feature_df)

        # Okay now run inference against our Features DataFrame
        my_prediction_df = my_endpoint.predict(my_feature_df)
        print(my_prediction_df)

        # Compute performance metrics for out test predictions
        if REGRESSION:
            my_metrics = my_endpoint.regression_metrics(my_target_column, my_prediction_df)
        else:
            my_metrics = my_endpoint.classification_metrics(my_target_column, my_prediction_df)
        print(my_metrics)

        # Capture the performance metrics for this Endpoint
        my_endpoint.capture_performance_metrics(
            my_feature_df, my_target_column, data_name=data_name, data_hash=data_hash, description=description
        )
