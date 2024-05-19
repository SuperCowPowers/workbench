"""ModelCore: SageWorks ModelCore Class"""

import time
from datetime import datetime
import urllib.parse
from typing import Union
from enum import Enum
import botocore

import pandas as pd
import awswrangler as wr
from urllib.parse import urlparse
from awswrangler.exceptions import NoFilesFound
from sagemaker import TrainingJobAnalytics
from sagemaker.model import Model as SagemakerModel

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.aws_utils import newest_files, pull_s3_data


# Enumerated Model Types
class ModelType(Enum):
    """Enumerated Types for SageWorks Model Types"""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    UNSUPERVISED = "unsupervised"
    TRANSFORMER = "transformer"
    UNKNOWN = "unknown"


class ModelCore(Artifact):
    """ModelCore: SageWorks ModelCore Class

    Common Usage:
        ```
        my_model = ModelCore(model_uuid)
        my_model.summary()
        my_model.details()
        ```
    """

    def __init__(
        self, model_uuid: str, force_refresh: bool = False, model_type: ModelType = None, legacy: bool = False
    ):
        """ModelCore Initialization
        Args:
            model_uuid (str): Name of Model in SageWorks.
            force_refresh (bool, optional): Force a refresh of the AWS Broker. Defaults to False.
            model_type (ModelType, optional): Set this for newly created Models. Defaults to None.
            legacy (bool, optional): Force load of legacy models. Defaults to False.
        """

        # Make sure the model name is valid
        if not legacy:
            self.ensure_valid_name(model_uuid, delimiter="-")

        # Call SuperClass Initialization
        super().__init__(model_uuid)

        # Grab an AWS Metadata Broker object and pull information for Models
        self.model_name = model_uuid
        aws_meta = self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=force_refresh)
        self.model_meta = aws_meta.get(self.model_name)
        if self.model_meta is None:
            self.log.important(f"Could not find model {self.model_name} within current visibility scope")
            self.latest_model = None
            self.model_type = ModelType.UNKNOWN
            return
        else:
            try:
                self.latest_model = self.model_meta[0]
                self.description = self.latest_model.get("ModelPackageDescription", "-")
                self.training_job_name = self._extract_training_job_name()
                if model_type:
                    self._set_model_type(model_type)
                else:
                    self.model_type = self._get_model_type()
            except (IndexError, KeyError):
                self.log.critical(f"Model {self.model_name} appears to be malformed. Delete and recreate it!")
                self.latest_model = None
                self.model_type = ModelType.UNKNOWN
                return

        # Set the Model Training S3 Path
        self.model_training_path = self.models_s3_path + "/training/" + self.model_name

        # Get our Endpoint Inference Path (might be None)
        self.endpoint_inference_path = self.get_endpoint_inference_path()

        # Call SuperClass Post Initialization
        super().__post_init__()

        # All done
        self.log.info(f"Model Initialized: {self.model_name}")

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self.model_meta = self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=True).get(self.model_name)
        self.latest_model = self.model_meta[0]
        self.description = self.latest_model.get("ModelPackageDescription", "-")
        self.training_job_name = self._extract_training_job_name()

    def exists(self) -> bool:
        """Does the model metadata exist in the AWS Metadata?"""
        if self.model_meta is None:
            self.log.debug(f"Model {self.model_name} not found in AWS Metadata!")
            return False
        return True

    def health_check(self) -> list[str]:
        """Perform a health check on this model
        Returns:
            list[str]: List of health issues
        """
        # Call the base class health check
        health_issues = super().health_check()

        # Model Type
        if self._get_model_type() == ModelType.UNKNOWN:
            health_issues.append("model_type_unknown")
        else:
            self.remove_health_tag("model_type_unknown")

        # Model Performance Metrics
        if self.performance_metrics() is None:
            health_issues.append("metrics_needed")
        else:
            self.remove_health_tag("metrics_needed")
        return health_issues

    def latest_model_object(self) -> SagemakerModel:
        """Return the latest AWS Sagemaker Model object for this SageWorks Model

        Returns:
           sagemaker.model.Model: AWS Sagemaker Model object
        """
        return SagemakerModel(
            model_data=self.model_package_arn(), sagemaker_session=self.sm_session, image_uri=self.model_image()
        )

    def list_inference_runs(self) -> list[str]:
        """List the inference runs for this model

        Returns:
            list[str]: List of inference run UUIDs
        """
        if self.endpoint_inference_path is None:
            return ["model_training"]  # Just the training run
        directories = wr.s3.list_directories(path=self.endpoint_inference_path + "/")
        inference_runs = [urlparse(directory).path.split("/")[-2] for directory in directories]

        # We're going to add the training to the front of the list
        inference_runs.insert(0, "model_training")
        return inference_runs

    def performance_metrics(self, capture_uuid: str = "latest") -> Union[pd.DataFrame, None]:
        """Retrieve the performance metrics for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid or "training" (default: "latest")
        Returns:
            pd.DataFrame: DataFrame of the Model Metrics

        Note:
            If a capture_uuid isn't specified this will try to return something reasonable
        """
        # Try to get the auto_capture 'training_holdout' or the training
        if capture_uuid == "latest":
            metrics_df = self.performance_metrics("training_holdout")
            return metrics_df if metrics_df is not None else self.performance_metrics("model_training")

        # Grab the metrics captured during model training (could return None)
        if capture_uuid == "model_training":
            metrics = self.sageworks_meta().get("sageworks_training_metrics")
            return pd.DataFrame.from_dict(metrics) if metrics else None

        else:  # Specific capture_uuid (could return None)
            s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_metrics.csv"
            metrics = pull_s3_data(s3_path, embedded_index=True)
            if metrics is not None:
                return metrics
            else:
                self.log.warning(f"Performance metrics {capture_uuid} not found for {self.model_name}!")
                return None

    def confusion_matrix(self, capture_uuid: str = "latest") -> Union[pd.DataFrame, None]:
        """Retrieve the confusion_matrix for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid or "training" (default: "latest")
        Returns:
            pd.DataFrame: DataFrame of the Confusion Matrix (might be None)
        """
        # Grab the metrics from the SageWorks Metadata (try inference first, then training)
        if capture_uuid == "latest":
            cm = self.sageworks_meta().get("sageworks_inference_cm")
            return cm if cm is not None else self.confusion_matrix("model_training")

        # Grab the confusion matrix captured during model training (could return None)
        if capture_uuid == "model_training":
            cm = self.sageworks_meta().get("sageworks_training_cm")
            return pd.DataFrame.from_dict(cm) if cm else None

        else:  # Specific capture_uuid
            s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_cm.csv"
            cm = pull_s3_data(s3_path, embedded_index=True)
            if cm is not None:
                return cm
            else:
                self.log.warning(f"Confusion Matrix {capture_uuid} not found for {self.model_name}!")
                return None

    def predictions(self, capture_uuid: str = "training_holdout") -> Union[pd.DataFrame, None]:
        """Retrieve the predictions for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid or "training" (default: "training_holdout")
        Returns:
            pd.DataFrame: DataFrame of the Predictions (might be None)
        """
        # Grab the metrics from the SageWorks Metadata (try inference first, then training)
        inference_preds = self.inference_predictions(capture_uuid)
        if inference_preds is not None:
            return inference_preds
        return self.validation_predictions()

    def set_input(self, input: str, force: bool = False):
        """Override: Set the input data for this artifact

        Args:
            input (str): Name of input for this artifact
            force (bool, optional): Force the input to be set (default: False)
        Note:
            We're going to not allow this to be used for Models
        """
        if not force:
            self.log.warning(f"Model {self.uuid}: Does not allow manual override of the input!")
            return

        # Okay we're going to allow this to be set
        self.log.important(f"{self.uuid}: Setting input to {input}...")
        self.log.important("Be careful with this! It breaks automatic provenance of the artifact!")
        self.upsert_sageworks_meta({"sageworks_input": input})

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

    def model_container_info(self) -> dict:
        """Container Info for the Latest Model Package"""
        return self.latest_model["ModelPackageDetails"]["InferenceSpecification"]["Containers"][0]

    def model_image(self) -> str:
        """Container Image for the Latest Model Package"""
        return self.model_container_info()["Image"]

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.latest_model["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.latest_model["CreationTime"]

    def register_endpoint(self, endpoint_name: str):
        """Add this endpoint to the set of registered endpoints for the model

        Args:
            endpoint_name (str): Name of the endpoint
        """
        self.log.important(f"Registering Endpoint {endpoint_name} with Model {self.uuid}...")
        registered_endpoints = set(self.sageworks_meta().get("sageworks_registered_endpoints", []))
        registered_endpoints.add(endpoint_name)
        self.upsert_sageworks_meta({"sageworks_registered_endpoints": list(registered_endpoints)})

        # A new endpoint means we need to refresh our inference path
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.endpoint_inference_path = self.get_endpoint_inference_path()

    def endpoints(self) -> list[str]:
        """Get the list of registered endpoints for this Model

        Returns:
            list[str]: List of registered endpoints
        """
        return self.sageworks_meta().get("sageworks_registered_endpoints", [])

    def get_endpoint_inference_path(self) -> str:
        """Get the S3 Path for the Inference Data"""

        # Look for any Registered Endpoints
        registered_endpoints = self.sageworks_meta().get("sageworks_registered_endpoints")

        # Note: We may have 0 to N endpoints, so we find the one with the most recent artifacts
        if registered_endpoints:
            endpoint_inference_base = self.endpoints_s3_path + "/inference/"
            endpoint_inference_paths = [endpoint_inference_base + e for e in registered_endpoints]
            return newest_files(endpoint_inference_paths, self.sm_session)
        else:
            self.log.warning(f"No registered endpoints found for {self.model_name}!")
            return None

    def set_target(self, target_column: str):
        """Set the target for this Model

        Args:
            target_column (str): Target column for this Model
        """
        self.upsert_sageworks_meta({"sageworks_model_target": target_column})

    def set_features(self, feature_columns: list[str]):
        """Set the features for this Model

        Args:
            feature_columns (list[str]): List of feature columns
        """
        self.upsert_sageworks_meta({"sageworks_model_features": feature_columns})

    def target(self) -> Union[str, None]:
        """Return the target for this Model (if supervised, else None)

        Returns:
            str: Target column for this Model (if supervised, else None)
        """
        return self.sageworks_meta().get("sageworks_model_target")  # Returns None if not found

    def features(self) -> Union[list[str], None]:
        """Return a list of features used for this Model

        Returns:
            list[str]: List of features used for this Model
        """
        return self.sageworks_meta().get("sageworks_model_features")  # Returns None if not found

    def class_labels(self) -> Union[list[str], None]:
        """Return the class labels for this Model (if it's a classifier)

        Returns:
            list[str]: List of class labels
        """
        if self.model_type == ModelType.CLASSIFIER:
            return self.sageworks_meta().get("class_labels")  # Returns None if not found
        else:
            return None

    def set_class_labels(self, labels: list[str]):
        """Return the class labels for this Model (if it's a classifier)

        Args:
            labels (list[str]): List of class labels
        """
        if self.model_type == ModelType.CLASSIFIER:
            self.upsert_sageworks_meta({"class_labels": labels})
        else:
            self.log.error(f"Model {self.model_name} is not a classifier!")

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
        details["pipeline"] = self.get_pipeline()
        details["model_type"] = self.model_type.value
        details["model_package_group_arn"] = self.group_arn()
        details["model_package_arn"] = self.model_package_arn()
        aws_meta = self.aws_meta()
        details["description"] = aws_meta.get("ModelPackageDescription", "-")
        details["version"] = aws_meta["ModelPackageVersion"]
        details["status"] = aws_meta["ModelPackageStatus"]
        details["approval_status"] = aws_meta["ModelApprovalStatus"]
        details["image"] = self.model_image().split("/")[-1]  # Shorten the image uri

        # Grab the inference and container info
        package_details = aws_meta["ModelPackageDetails"]
        inference_spec = package_details["InferenceSpecification"]
        container_info = self.model_container_info()
        details["framework"] = container_info.get("Framework", "unknown")
        details["framework_version"] = container_info.get("FrameworkVersion", "unknown")
        details["inference_types"] = inference_spec["SupportedRealtimeInferenceInstanceTypes"]
        details["transform_types"] = inference_spec["SupportedTransformInstanceTypes"]
        details["content_types"] = inference_spec["SupportedContentTypes"]
        details["response_types"] = inference_spec["SupportedResponseMIMETypes"]
        details["model_metrics"] = self.performance_metrics()
        if self.model_type == ModelType.CLASSIFIER:
            details["confusion_matrix"] = self.confusion_matrix()
            details["predictions"] = None
        else:
            details["confusion_matrix"] = None
            details["predictions"] = self.predictions()

        # Grab the inference metadata
        details["inference_meta"] = self.inference_metadata()

        # Cache the details
        self.data_storage.set(storage_key, details)

        # Return the details
        return details

    # Pipeline for this model
    def get_pipeline(self) -> str:
        """Get the pipeline for this model"""
        return self.sageworks_meta().get("sageworks_pipeline")

    def set_pipeline(self, pipeline: str):
        """Set the pipeline for this model

        Args:
            pipeline (str): Pipeline that was used to create this model
        """
        self.upsert_sageworks_meta({"sageworks_pipeline": pipeline})

    def expected_meta(self) -> list[str]:
        """Metadata we expect to see for this Model when it's ready
        Returns:
            list[str]: List of expected metadata keys
        """
        # Our current list of expected metadata, we can add to this as needed
        return ["sageworks_status", "sageworks_training_metrics", "sageworks_training_cm"]

    def is_model_unknown(self) -> bool:
        """Is the Model Type unknown?"""
        return self.model_type == ModelType.UNKNOWN

    def _determine_model_type(self):
        """Internal: Determine the Model Type"""
        model_type = input("Model Type? (classifier, regressor, unsupervised, transformer): ")
        if model_type == "classifier":
            self._set_model_type(ModelType.CLASSIFIER)
        elif model_type == "regressor":
            self._set_model_type(ModelType.REGRESSOR)
        elif model_type == "unsupervised":
            self._set_model_type(ModelType.UNSUPERVISED)
        elif model_type == "transformer":
            self._set_model_type(ModelType.TRANSFORMER)
        else:
            self.log.warning(f"Unknown Model Type {model_type}!")
            self._set_model_type(ModelType.UNKNOWN)

    def onboard(self, ask_everything=False) -> bool:
        """This is an interactive method that will onboard the Model (make it ready)

        Args:
            ask_everything (bool, optional): Ask for all the details. Defaults to False.

        Returns:
            bool: True if the Model is successfully onboarded, False otherwise
        """
        # Set the status to onboarding
        self.set_status("onboarding")

        # Determine the Model Type
        while self.is_model_unknown():
            self._determine_model_type()

        # Determine the Target Column (can be None)
        target_column = self.target()
        if target_column is None or ask_everything:
            target_column = input("Target Column? (for unsupervised/transformer just type None): ")
            if target_column in ["None", "none", ""]:
                target_column = None

        # Determine the Feature Columns
        feature_columns = self.features()
        if feature_columns is None or ask_everything:
            feature_columns = input("Feature Columns? (use commas): ")
            feature_columns = [e.strip() for e in feature_columns.split(",")]
            if feature_columns in [["None"], ["none"], [""]]:
                feature_columns = None

        # Registered Endpoints?
        endpoints = self.endpoints()
        if not endpoints or ask_everything:
            endpoints = input("Register Endpoints? (use commas for multiple): ")
            endpoints = [e.strip() for e in endpoints.split(",")]
            if endpoints in [["None"], ["none"], [""]]:
                endpoints = None

        # Model Owner?
        owner = self.get_owner()
        if owner in [None, "unknown"] or ask_everything:
            owner = input("Model Owner: ")
            if owner in ["None", "none", ""]:
                owner = "unknown"

        # Now that we have all the details, let's onboard the Model with all the args
        return self.onboard_with_args(self.model_type, target_column, feature_columns, endpoints, owner)

    def onboard_with_args(
        self,
        model_type: ModelType,
        target_column: str = None,
        feature_list: list = None,
        endpoints: list = None,
        owner: str = None,
    ) -> bool:
        """Onboard the Model with the given arguments

        Args:
            model_type (ModelType): Model Type
            target_column (str): Target Column
            feature_list (list): List of Feature Columns
            endpoints (list, optional): List of Endpoints. Defaults to None.
            owner (str, optional): Model Owner. Defaults to None.
        Returns:
            bool: True if the Model is successfully onboarded, False otherwise
        """
        # Set the status to onboarding
        self.set_status("onboarding")

        # Set All the Details
        self._set_model_type(model_type)
        if target_column:
            self.set_target(target_column)
        if feature_list:
            self.set_features(feature_list)
        if endpoints:
            for endpoint in endpoints:
                self.register_endpoint(endpoint)
        if owner:
            self.set_owner(owner)

        # Load the training metrics and inference metrics
        self._load_training_metrics()
        self._load_inference_metrics()
        self._load_inference_cm()

        # Remove the needs_onboard tag
        self.remove_health_tag("needs_onboard")

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details(recompute=True)
        self.set_status("ready")
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

        # Delete any training artifacts
        s3_delete_path = f"{self.model_training_path}/"
        self.log.info(f"Deleting Training S3 Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=self.boto_session)

        # Delete any data in the Cache
        for key in self.data_storage.list_subkeys(f"model:{self.uuid}:"):
            self.log.info(f"Deleting Cache Key {key}...")
            self.data_storage.delete(key)

    def _set_model_type(self, model_type: ModelType):
        """Internal: Set the Model Type for this Model"""
        self.model_type = model_type
        self.upsert_sageworks_meta({"sageworks_model_type": self.model_type.value})
        self.remove_health_tag("model_type_unknown")

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
            self.log.warning(f"Could not determine model type for {self.model_name}!")
            return ModelType.UNKNOWN

    def _load_training_metrics(self):
        """Internal: Retrieve the training metrics and Confusion Matrix for this model
                     and load the data into the SageWorks Metadata

        Notes:
            This may or may not exist based on whether we have access to TrainingJobAnalytics
        """
        try:
            df = TrainingJobAnalytics(training_job_name=self.training_job_name).dataframe()
            if df.empty:
                self.log.warning(f"No training job metrics found for {self.training_job_name}")
                self.upsert_sageworks_meta({"sageworks_training_metrics": None, "sageworks_training_cm": None})
                return
            if self.model_type == ModelType.REGRESSOR:
                if "timestamp" in df.columns:
                    df = df.drop(columns=["timestamp"])

                # We're going to pivot the DataFrame to get the desired structure
                reg_metrics_df = df.set_index("metric_name").T

                # Store and return the metrics in the SageWorks Metadata
                self.upsert_sageworks_meta(
                    {"sageworks_training_metrics": reg_metrics_df.to_dict(), "sageworks_training_cm": None}
                )
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
            self.upsert_sageworks_meta(
                {"sageworks_training_metrics": metrics_df.to_dict(), "sageworks_training_cm": cm_df.to_dict()}
            )

    def _load_inference_metrics(self, capture_uuid: str = "training_holdout"):
        """Internal: Retrieve the inference model metrics for this model
                     and load the data into the SageWorks Metadata

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "training_holdout")
        Notes:
            This may or may not exist based on whether an Endpoint ran Inference
        """
        s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_metrics.csv"
        inference_metrics = pull_s3_data(s3_path)

        # Store data into the SageWorks Metadata
        metrics_storage = None if inference_metrics is None else inference_metrics.to_dict("records")
        self.upsert_sageworks_meta({"sageworks_inference_metrics": metrics_storage})

    def _load_inference_cm(self, capture_uuid: str = "training_holdout"):
        """Internal: Pull the inference Confusion Matrix for this model
                     and load the data into the SageWorks Metadata

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "training_holdout")

        Returns:
            pd.DataFrame: DataFrame of the inference Confusion Matrix (might be None)

        Notes:
            This may or may not exist based on whether an Endpoint ran Inference
        """
        s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_cm.csv"
        inference_cm = pull_s3_data(s3_path, embedded_index=True)

        # Store data into the SageWorks Metadata
        cm_storage = None if inference_cm is None else inference_cm.to_dict("records")
        self.upsert_sageworks_meta({"sageworks_inference_cm": cm_storage})

    def inference_metadata(self, capture_uuid: str = "training_holdout") -> Union[pd.DataFrame, None]:
        """Retrieve the inference metadata for this model

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "training_holdout")

        Returns:
            dict: Dictionary of the inference metadata (might be None)
        Notes:
            Basically when Endpoint inference was run, name of the dataset, the MD5, etc
        """
        # Sanity check the inference path (which may or may not exist)
        if self.endpoint_inference_path is None:
            return None

        # Check for model_training capture_uuid
        if capture_uuid == "model_training":
            # Create a DataFrame with the training metadata
            meta_df = pd.DataFrame(
                [
                    {
                        "name": "AWS Training Capture",
                        "data_hash": "N/A",
                        "num_rows": "-",
                        "description": "-",
                    }
                ]
            )
            return meta_df

        # Pull the inference metadata
        try:
            s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_meta.json"
            return wr.s3.read_json(s3_path)
        except NoFilesFound:
            self.log.info(f"Could not find model inference meta at {s3_path}...")
            return None

    def inference_predictions(self, capture_uuid: str = "training_holdout") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: training_holdout)

        Returns:
            pd.DataFrame: DataFrame of the Captured Predictions (might be None)
        """
        self.log.important(f"Grabbing {capture_uuid} predictions for {self.model_name}...")
        s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_predictions.csv"
        return pull_s3_data(s3_path)

    def validation_predictions(self) -> Union[pd.DataFrame, None]:
        """Internal: Retrieve the captured prediction results for this model

        Returns:
            pd.DataFrame: DataFrame of the Captured Validation Predictions (might be None)
        """
        self.log.important(f"Grabbing Validation Predictions for {self.model_name}...")
        s3_path = f"{self.model_training_path}/validation_predictions.csv"
        df = pull_s3_data(s3_path)
        return df

    def _extract_training_job_name(self) -> Union[str, None]:
        """Internal: Extract the training job name from the ModelDataUrl"""
        try:
            model_data_url = self.model_container_info()["ModelDataUrl"]
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

        # Convert the values in cm_df to integers
        cm_df = cm_df.astype(int)

        return metrics_df, cm_df

    def shapley_values(self, capture_uuid: str = "training_holdout") -> Union[list[pd.DataFrame], pd.DataFrame, None]:
        """Retrieve the Shapely values for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: training_holdout)

        Returns:
            pd.DataFrame: Dataframe of the shapley values for the prediction dataframe

        Notes:
            This may or may not exist based on whether an Endpoint ran Shapley
        """

        # Sanity check the inference path (which may or may not exist)
        if self.endpoint_inference_path is None:
            return None

        # Construct the S3 path for the Shapley values
        shapley_s3_path = f"{self.endpoint_inference_path}/{capture_uuid}"

        # Multiple CSV if classifier
        if self.model_type == ModelType.CLASSIFIER:
            # CSVs for shap values are indexed by prediction class
            # Because we don't know how many classes there are, we need to search through
            # a list of S3 objects in the parent folder
            s3_paths = wr.s3.list_objects(shapley_s3_path)
            return [pull_s3_data(f) for f in s3_paths if "inference_shap_values" in f]

        # One CSV if regressor
        if self.model_type == ModelType.REGRESSOR:
            s3_path = f"{shapley_s3_path}/inference_shap_values.csv"
            return pull_s3_data(s3_path)


if __name__ == "__main__":
    """Exercise the ModelCore Class"""

    # Grab a ModelCore object and pull some information from it
    my_model = ModelCore("abalone-regression")

    # Call the various methods

    # Let's do a check/validation of the Model
    print(f"Model Check: {my_model.exists()}")

    # Make sure the model is 'ready'
    my_model.onboard(interactive=False)

    # Get the ARN of the Model Group
    print(f"Model Group ARN: {my_model.group_arn()}")
    print(f"Model Package ARN: {my_model.arn()}")

    # Get the tags associated with this Model
    print(f"Tags: {my_model.get_tags()}")

    # Get the SageWorks metadata associated with this Model
    print(f"SageWorks Meta: {my_model.sageworks_meta()}")

    # Get creation time
    print(f"Created: {my_model.created()}")

    # Get training job name
    print(f"Training Job: {my_model.training_job_name}")

    # List any inference runs
    print(f"Inference Runs: {my_model.list_inference_runs()}")

    # Get any captured metrics from the training job
    print("Model Metrics:")
    print(my_model.performance_metrics())

    print("Confusion Matrix: (might be None)")
    print(my_model.confusion_matrix())

    # Grab our regression predictions from S3
    print("Captured Predictions: (might be None)")
    print(my_model.predictions())

    # Grab our Shapley values from S3
    print("Shapley Values: (might be None)")
    print(my_model.shapley_values())

    # Get the SageWorks metadata associated with this Model
    print(f"SageWorks Meta: {my_model.sageworks_meta()}")

    # Get the latest model object (sagemaker.model.Model)
    sagemaker_model = my_model.latest_model_object()
    print(f"Latest Model Object: {my_model.latest_model_object()}")

    # Get the Class Labels (if it's a classifier)
    my_model = ModelCore("wine-classification")
    print(f"Class Labels: {my_model.class_labels()}")
    my_model.set_class_labels(["red", "white"])
    print(f"Class Labels: {my_model.class_labels()}")

    # Delete the Model
    # my_model.delete()
