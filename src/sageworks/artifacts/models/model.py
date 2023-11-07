"""Model: SageWorks Model Class"""
from datetime import datetime
import urllib.parse
from typing import Union
from enum import Enum
import botocore

import pandas as pd
import awswrangler as wr
from awswrangler.exceptions import NoFilesFound
from sagemaker import TrainingJobAnalytics

# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


# Enumerated Model Types
class ModelType(Enum):
    """Enumerated Types for SageWorks Model Types"""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    UNKNOWN = "unknown"


class Model(Artifact):
    """Model: SageWorks Model Class

    Common Usage:
        my_model = Model(model_uuid)
        my_model.summary()
        my_model.details()
    """

    def __init__(self, model_uuid: str, force_refresh: bool = False, model_type: ModelType = None):
        """Model Initialization
        Args:
            model_uuid (str): Name of Model in SageWorks.
            force_refresh (bool, optional): Force a refresh of the AWS Broker. Defaults to False.
            model_type (ModelType, optional): Set this for newly created Models. Defaults to None.
        """
        # Call SuperClass Initialization
        super().__init__(model_uuid)

        # Set the Model Training and Inference S3 Paths
        self.model_training_path = self.models_s3_path + "/training"
        self.model_inference_path = self.models_s3_path + "/inference"

        # Grab an AWS Metadata Broker object and pull information for Models
        self.model_name = model_uuid
        aws_meta = self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=force_refresh)
        self.model_meta = aws_meta.get(self.model_name)
        if self.model_meta is None:
            self.log.important(f"Could not find model {self.model_name} within current visibility scope")
            self.latest_model = None
            self.model_type = ModelType.UNKNOWN
        else:
            try:
                self.latest_model = self.model_meta[0]
                self.description = self.latest_model["ModelPackageDescription"]
                self.training_job_name = self._extract_training_job_name()
                if model_type:
                    self._set_model_type(model_type)
                else:
                    self.model_type = self._get_model_type()
            except (IndexError, KeyError):
                self.log.critical(f"Model {self.model_name} appears to be malformed. Delete and recreate it!")
                self.latest_model = None
                self.model_type = ModelType.UNKNOWN

        # All done
        self.log.info(f"Model Initialized: {self.model_name}")

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self.model_meta = self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=True).get(self.model_name)
        self.latest_model = self.model_meta[0]
        self.description = self.latest_model["ModelPackageDescription"]
        self.training_job_name = self._extract_training_job_name()

    def exists(self) -> bool:
        """Does the model metadata exist in the AWS Metadata?"""
        if self.model_meta is None:
            self.log.info(f"Model {self.model_name} not found in AWS Metadata!")
            return False
        return True

    def _set_model_type(self, model_type: ModelType):
        """Internal: Set the Model Type for this Model"""
        self.model_type = model_type
        self.upsert_sageworks_meta({"sageworks_model_type": self.model_type.value})

    def _get_model_type(self) -> ModelType:
        """Internal: Query the SageWorks Metadata to get the model type
        Returns:
            ModelType: The ModelType of this Model
        Notes:
            This is an internal method that should not be called directly
            Use the model_type attribute instead
        """
        model_type = self.sageworks_meta().get("sageworks_model_type")
        if model_type and model_type != "unknown":
            return ModelType(model_type)
        else:
            self.log.critical(f"Could not determine model type for {self.model_name}!")
            return ModelType.UNKNOWN

    def model_metrics(self) -> Union[pd.DataFrame, None]:
        """Retrieve the training metrics for this model
        Returns:
            pd.DataFrame: DataFrame of the Model Metrics
        """
        # Grab the metrics from the SageWorks Metadata (try inference first, then training)
        metrics = self._pull_inference_metrics()
        if metrics is not None:
            return metrics
        metrics = self.sageworks_meta().get("sageworks_training_metrics")
        return pd.DataFrame.from_dict(metrics) if metrics else None

    def confusion_matrix(self) -> Union[pd.DataFrame, None]:
        """Retrieve the confusion_matrix for this model
        Returns:
            pd.DataFrame: DataFrame of the Confusion Matrix (might be None)
        """
        # Grab the confusion matrix from the SageWorks Metadata
        cm = self._pull_inference_cm()
        if cm is not None:
            return cm
        cm = self.sageworks_meta().get("sageworks_training_cm")
        return pd.DataFrame.from_dict(cm) if cm else None

    def regression_predictions(self) -> Union[pd.DataFrame, None]:
        """Retrieve the regression based predictions for this model
        Returns:
            pd.DataFrame: DataFrame of the Regression based Predictions (might be None)
        """

        # Pull the regression predictions, try first from inference, then from training
        s3_path = f"{self.model_inference_path}/{self.model_name}/inference_predictions.csv"
        df = self._pull_s3_model_artifacts(s3_path)
        if df is not None:
            return df
        else:
            s3_path = f"{self.model_training_path}/{self.model_name}/validation_predictions.csv"
            df = self._pull_s3_model_artifacts(s3_path)
            return df

    def _pull_inference_metadata(self) -> Union[pd.DataFrame, None]:
        """Internal: Retrieve the inference metadata for this model
        Returns:
            dict: Dictionary of the inference metadata (might be None)
        Notes:
            Basically when the inference was run, name of the dataset, the MD5, etc
        """
        s3_path = f"{self.model_inference_path}/{self.model_name}/inference_meta.json"
        try:
            return wr.s3.read_json(s3_path)
        except NoFilesFound:
            self.log.info(f"Could not find model artifact at {s3_path}...")
            return None

    def _pull_inference_metrics(self) -> Union[pd.DataFrame, None]:
        """Internal: Retrieve the inference model metrics for this model
        Returns:
            pd.DataFrame: DataFrame of the inference model metrics (might be None)
        """
        s3_path = f"{self.model_inference_path}/{self.model_name}/inference_metrics.csv"
        return self._pull_s3_model_artifacts(s3_path)

    def _pull_inference_cm(self) -> Union[pd.DataFrame, None]:
        """Internal: Retrieve the inference Confusion Matrix for this model
        Returns:
            pd.DataFrame: DataFrame of the inference Confusion Matrix (might be None)
        """
        s3_path = f"{self.model_inference_path}/{self.model_name}/inference_cm.csv"
        return self._pull_s3_model_artifacts(s3_path, embedded_index=True)

    def _pull_s3_model_artifacts(self, s3_path, embedded_index=False) -> Union[pd.DataFrame, None]:
        """Internal: Helper method to pull Model Artifact data from S3 storage
        Args:
            s3_path (str): S3 Path to the Model Artifact
            embedded_index (bool, optional): Is the index embedded in the CSV? Defaults to False.
        Returns:
            pd.DataFrame: DataFrame of the Model Artifact (metrics, CM, regression_preds) (might be None)
        """

        # Pull the CSV file from S3
        try:
            if embedded_index:
                df = wr.s3.read_csv(s3_path, index_col=0)
            else:
                df = wr.s3.read_csv(s3_path)
            return df
        except NoFilesFound:
            self.log.info(f"Could not find model artifact at {s3_path}...")
            return None

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        return 0.0

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.latest_model

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for the Model Package Group"""
        return self.group_arn()

    def group_arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for the Model Package Group"""
        return self.latest_model["ModelPackageGroupArn"]

    def model_package_arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for the Model Package (within the Group)"""
        return self.latest_model["ModelPackageArn"]

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.latest_model["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.latest_model["CreationTime"]

    def details(self, recompute=False) -> dict:
        """Additional Details about this Model
        Args:
            recompute (bool, optional): Recompute the details (default: False)
        Returns:
            dict: Dictionary of details about this Model
        """

        # Check if we have cached version of the Model Details
        storage_key = f"model:{self.uuid}:details"
        cached_details = self.data_storage.get(storage_key)
        if cached_details and not recompute:
            return cached_details

        self.log.info("Recomputing Model Details...")
        details = self.summary()
        details["model_type"] = self.model_type.value
        details["model_package_group_arn"] = self.group_arn()
        details["model_package_arn"] = self.model_package_arn()
        aws_meta = self.aws_meta()
        details["description"] = aws_meta["ModelPackageDescription"]
        details["status"] = aws_meta["ModelPackageStatus"]
        details["approval_status"] = aws_meta["ModelApprovalStatus"]
        package_details = aws_meta["ModelPackageDetails"]
        inference_spec = package_details["InferenceSpecification"]
        container = inference_spec["Containers"][0]
        image_short = container["Image"].split("/")[-1]
        details["image"] = image_short
        details["framework"] = container.get("Framework", "unknown")
        details["framework_version"] = container.get("FrameworkVersion", "unknown")
        details["inference_types"] = inference_spec["SupportedRealtimeInferenceInstanceTypes"]
        details["transform_types"] = inference_spec["SupportedTransformInstanceTypes"]
        details["content_types"] = inference_spec["SupportedContentTypes"]
        details["response_types"] = inference_spec["SupportedResponseMIMETypes"]
        details["model_metrics"] = self.model_metrics()
        if self.model_type == ModelType.CLASSIFIER:
            details["confusion_matrix"] = self.confusion_matrix()
            details["regression_predictions"] = None
        else:
            details["confusion_matrix"] = None
            details["regression_predictions"] = self.regression_predictions()

        # Grab the inference metadata
        details["inference_meta"] = self._pull_inference_metadata()

        # Cache the details
        self.data_storage.set(storage_key, details)

        # Return the details
        return details

    def expected_meta(self) -> list[str]:
        """Metadata we expect to see for this Artifact when it's ready
        Returns:
            list[str]: List of expected metadata keys
        """

        # If an artifact has additional expected metadata override this method
        return ["sageworks_status", "sageworks_training_metrics", "sageworks_training_cm"]

    def make_ready(self) -> bool:
        """This is a BLOCKING method that will wait until the Model is ready"""
        self._pull_training_job_metrics()
        self.details(recompute=True)
        self.set_status("ready")
        self.refresh_meta()
        return True

    def delete(self):
        """Delete the Model Packages and the Model Group"""

        # If we don't have meta then the model probably doesn't exist
        if self.model_meta is None:
            self.log.info(f"Model {self.model_name} doesn't appear to exist...")
            return

        # First delete the Model Packages within the Model Group
        for model in self.model_meta:
            self.log.info(f"Deleting Model Package {model['ModelPackageArn']}...")
            self.sm_client.delete_model_package(ModelPackageName=model["ModelPackageArn"])

        # Delete the Model Package Group
        self.log.info(f"Deleting Model Group {self.model_name}...")
        self.sm_client.delete_model_package_group(ModelPackageGroupName=self.model_name)

        # Delete any inference artifacts
        s3_delete_path = f"{self.model_inference_path}/{self.model_name}"
        self.log.info(f"Deleting Training S3 Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=self.boto_session)

        # Delete any training artifacts
        s3_delete_path = f"{self.model_training_path}/{self.model_name}"
        self.log.info(f"Deleting Inference S3 Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=self.boto_session)

        # Delete any data in the Cache
        for key in self.data_storage.list_subkeys(f"model:{self.uuid}"):
            self.log.info(f"Deleting Cache Key {key}...")
            self.data_storage.delete(key)

    def _pull_training_job_metrics(self, force_pull=False):
        """Internal: Grab any captured metrics from the training job for this model
        Args:
            force_pull (bool, optional): Force a pull from TrainingJobAnalytics. Defaults to False.
        """

        # First check if we have already computed the various metrics
        model_metrics = self.sageworks_meta().get("sageworks_training_metrics")
        if self.model_type == ModelType.REGRESSOR:
            if model_metrics and not force_pull:
                return

        # For classifiers, we need to pull the confusion matrix as well
        cm = self.sageworks_meta().get("sageworks_training_cm")
        if model_metrics and cm and not force_pull:
            return

        # We don't have them, so go and grab the training job metrics
        self.log.info(f"Pulling training job metrics for {self.training_job_name}...")
        try:
            df = TrainingJobAnalytics(training_job_name=self.training_job_name).dataframe()
            if self.model_type == ModelType.REGRESSOR:
                if "timestamp" in df.columns:
                    df = df.drop(columns=["timestamp"])

                # Store and return the metrics in the SageWorks Metadata
                df = df.round(3)
                self.upsert_sageworks_meta({"sageworks_training_metrics": df.to_dict(), "sageworks_training_cm": None})
                return
        except (KeyError, botocore.exceptions.ClientError):
            self.log.warning(f"No training job metrics found for {self.training_job_name}")
            # Store and return the metrics in the SageWorks Metadata
            self.upsert_sageworks_meta({"sageworks_training_metrics": None, "sageworks_training_cm": None})
            return

        # We need additional processing for classification metrics
        if self.model_type == ModelType.CLASSIFIER:
            metrics_df, cm_df = self._process_classification_metrics(df)

            # Store and return the metrics in the SageWorks Metadata
            metrics_df = metrics_df.round(3)
            cm_df = cm_df.round(3)
            self.upsert_sageworks_meta(
                {"sageworks_training_metrics": metrics_df.to_dict(), "sageworks_training_cm": cm_df.to_dict()}
            )

    def _extract_training_job_name(self) -> Union[str, None]:
        """Internal: Extract the training job name from the ModelDataUrl"""
        try:
            container = self.latest_model["ModelPackageDetails"]["InferenceSpecification"]["Containers"][0]
            model_data_url = container["ModelDataUrl"]
            parsed_url = urllib.parse.urlparse(model_data_url)
            training_job_name = parsed_url.path.lstrip("/").split("/")[0]
            return training_job_name
        except KeyError:
            self.log.warning(f"Could not extract training job name from {model_data_url}")
            return None

    @staticmethod
    def _process_classification_metrics(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Internal: Process classification metrics into a more reasonable format
        Args:
            df (pd.DataFrame): DataFrame of training metrics
        Returns:
            (pd.DataFrame, pd.DataFrame): Tuple of DataFrames. Metrics and confusion matrix
        """
        # Split into two DataFrames based on 'metric_name'
        metrics_df = df[df["metric_name"].str.startswith("Metrics:")].copy()
        cm_df = df[df["metric_name"].str.startswith("ConfusionMatrix:")].copy()

        # Split the 'metric_name' into different parts
        metrics_df["class"] = metrics_df["metric_name"].str.split(":").str[1]
        metrics_df["metric_type"] = metrics_df["metric_name"].str.split(":").str[2]

        # Pivot the DataFrame to get the desired structure
        metrics_df = metrics_df.pivot(index="class", columns="metric_type", values="value").reset_index()
        metrics_df = metrics_df.rename_axis(None, axis=1)

        # Now process the confusion matrix
        cm_df["row_class"] = cm_df["metric_name"].str.split(":").str[1]
        cm_df["col_class"] = cm_df["metric_name"].str.split(":").str[2]

        # Pivot the DataFrame to create a form suitable for the heatmap
        cm_df = cm_df.pivot(index="row_class", columns="col_class", values="value")

        # Normalize the rows between 0 and 1
        cm_df = cm_df.div(cm_df.sum(axis=1), axis=0)

        return metrics_df, cm_df


if __name__ == "__main__":
    """Exercise the Model Class"""

    # Grab a Model object and pull some information from it
    my_model = Model("abalone-regression")

    # Call the various methods

    # Let's do a check/validation of the Model
    print(f"Model Check: {my_model.exists()}")

    # Make sure the model is 'ready'
    my_model.make_ready()

    # Get the ARN of the Model Group
    print(f"Model Group ARN: {my_model.group_arn()}")
    print(f"Model Package ARN: {my_model.arn()}")

    # Get the tags associated with this Model
    print(f"Tags: {my_model.sageworks_tags()}")

    # Get the SageWorks metadata associated with this Model
    print(f"SageWorks Meta: {my_model.sageworks_meta()}")

    # Get creation time
    print(f"Created: {my_model.created()}")

    # Get training job name
    print(f"Training Job: {my_model.training_job_name}")

    # Get any captured metrics from the training job
    print("Training Metrics:")
    print(my_model.model_metrics())

    print("Confusion Matrix: (might be None)")
    print(my_model.confusion_matrix())

    # Grab our regression predictions from S3
    print("Regression Predictions: (might be None)")
    print(my_model.regression_predictions())

    # Test Large Metadata
    # my_model.upsert_sageworks_meta({"sageworks_large_meta": {"large_x": "x" * 200, "large_y": "y" * 200}})

    # Test Deleting specific metadata
    # my_model.delete_metadata(["sageworks_large_meta"])

    # Get the SageWorks metadata associated with this Model
    print(f"SageWorks Meta: {my_model.sageworks_meta()}")

    # Delete the Model
    # my_model.delete()
