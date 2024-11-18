"""ModelCore: SageWorks ModelCore Class"""

import time
from datetime import datetime
import urllib.parse
from typing import Union
from enum import Enum
import botocore
from botocore.exceptions import ClientError

import pandas as pd
import awswrangler as wr
from urllib.parse import urlparse
from awswrangler.exceptions import NoFilesFound
from sagemaker import TrainingJobAnalytics
from sagemaker.model import Model as SagemakerModel

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.utils.aws_utils import newest_path, pull_s3_data


class ModelType(Enum):
    """Enumerated Types for SageWorks Model Types"""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"
    TRANSFORMER = "transformer"
    PROJECTION = "projection"
    UNSUPERVISED = "unsupervised"
    QUANTILE_REGRESSOR = "quantile_regressor"
    DETECTOR = "detector"
    UNKNOWN = "unknown"


class InferenceImage:
    """Class for retrieving locked Scikit-Learn inference images"""

    image_uris = {
        ("us-east-1", "sklearn", "1.2.1"): (
            "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
            "sagemaker-scikit-learn@sha256:ed242e33af079f334972acd2a7ddf74d13310d3c9a0ef3a0e9b0429ccc104dcd"
        ),
        ("us-east-2", "sklearn", "1.2.1"): (
            "257758044811.dkr.ecr.us-east-2.amazonaws.com/"
            "sagemaker-scikit-learn@sha256:ed242e33af079f334972acd2a7ddf74d13310d3c9a0ef3a0e9b0429ccc104dcd"
        ),
        ("us-west-1", "sklearn", "1.2.1"): (
            "746614075791.dkr.ecr.us-west-1.amazonaws.com/"
            "sagemaker-scikit-learn@sha256:ed242e33af079f334972acd2a7ddf74d13310d3c9a0ef3a0e9b0429ccc104dcd"
        ),
        ("us-west-2", "sklearn", "1.2.1"): (
            "246618743249.dkr.ecr.us-west-2.amazonaws.com/"
            "sagemaker-scikit-learn@sha256:ed242e33af079f334972acd2a7ddf74d13310d3c9a0ef3a0e9b0429ccc104dcd"
        ),
    }

    @classmethod
    def get_image_uri(cls, region, framework, version):
        key = (region, framework, version)
        if key in cls.image_uris:
            return cls.image_uris[key]
        else:
            raise ValueError(
                f"No matching image found for region: {region}, framework: {framework}, version: {version}"
            )


class ModelCore(Artifact):
    """ModelCore: SageWorks ModelCore Class

    Common Usage:
        ```python
        my_model = ModelCore(model_uuid)
        my_model.summary()
        my_model.details()
        ```
    """

    def __init__(self, model_uuid: str, model_type: ModelType = None, **kwargs):
        """ModelCore Initialization
        Args:
            model_uuid (str): Name of Model in SageWorks.
            model_type (ModelType, optional): Set this for newly created Models. Defaults to None.
            **kwargs: Additional keyword arguments
        """

        # Make sure the model name is valid
        self.is_name_valid(model_uuid, delimiter="-", lower_case=False)

        # Call SuperClass Initialization
        super().__init__(model_uuid, **kwargs)

        # Initialize our class attributes
        self.latest_model = None
        self.model_type = ModelType.UNKNOWN
        self.model_training_path = None
        self.endpoint_inference_path = None

        # Grab an Cloud Platform Meta object and pull information for this Model
        self.model_name = model_uuid
        self.model_meta = self.meta.model(self.model_name)
        if self.model_meta is None:
            self.log.warning(f"Could not find model {self.model_name} within current visibility scope")
            return
        else:
            # Is this a model package group without any models?
            if len(self.model_meta["ModelPackageList"]) == 0:
                self.log.warning(f"Model Group {self.model_name} has no Model Packages!")
                self.latest_model = None
                self.add_health_tag("model_not_found")
                return
            try:
                self.latest_model = self.model_meta["ModelPackageList"][0]
                self.description = self.latest_model.get("ModelPackageDescription", "-")
                self.training_job_name = self._extract_training_job_name()
                if model_type:
                    self._set_model_type(model_type)
                else:
                    self.model_type = self._get_model_type()
            except (IndexError, KeyError):
                self.log.critical(f"Model {self.model_name} appears to be malformed. Delete and recreate it!")
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
        self.model_meta = self.meta.model(self.model_name)
        self.latest_model = self.model_meta["ModelPackageList"][0]
        self.description = self.latest_model.get("ModelPackageDescription", "-")
        self.training_job_name = self._extract_training_job_name()

    def exists(self) -> bool:
        """Does the model metadata exist in the AWS Metadata?"""
        if self.model_meta is None:
            self.log.info(f"Model {self.model_name} not found in AWS Metadata!")
            return False
        return True

    def health_check(self) -> list[str]:
        """Perform a health check on this model
        Returns:
            list[str]: List of health issues
        """
        # Call the base class health check
        health_issues = super().health_check()

        # Check if the model exists
        if self.latest_model is None:
            health_issues.append("model_not_found")

        # Model Type
        if self._get_model_type() == ModelType.UNKNOWN:
            health_issues.append("model_type_unknown")
        else:
            self.remove_health_tag("model_type_unknown")

        # Model Performance Metrics
        needs_metrics = self.model_type in {ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR, ModelType.CLASSIFIER}
        if needs_metrics and self.get_inference_metrics() is None:
            health_issues.append("metrics_needed")
        else:
            self.remove_health_tag("metrics_needed")

        # Endpoint
        if not self.endpoints():
            health_issues.append("no_endpoint")
        else:
            self.remove_health_tag("no_endpoint")
        return health_issues

    def latest_model_object(self) -> SagemakerModel:
        """Return the latest AWS Sagemaker Model object for this SageWorks Model

        Returns:
           sagemaker.model.Model: AWS Sagemaker Model object
        """
        return SagemakerModel(
            model_data=self.model_package_arn(), sagemaker_session=self.sm_session, image_uri=self.container_image()
        )

    def list_inference_runs(self) -> list[str]:
        """List the inference runs for this model

        Returns:
            list[str]: List of inference runs
        """

        # Check if we have a model (if not return empty list)
        if self.latest_model is None:
            return []

        # Check if we have model training metrics in our metadata
        have_model_training = True if self.sageworks_meta().get("sageworks_training_metrics") else False

        # Now grab the list of directories from our inference path
        inference_runs = []
        if self.endpoint_inference_path:
            directories = wr.s3.list_directories(path=self.endpoint_inference_path + "/")
            inference_runs = [urlparse(directory).path.split("/")[-2] for directory in directories]

        # We're going to add the model training to the end of the list
        if have_model_training:
            inference_runs.append("model_training")
        return inference_runs

    def delete_inference_run(self, inference_run_uuid: str):
        """Delete the inference run for this model

        Args:
            inference_run_uuid (str): UUID of the inference run
        """
        if inference_run_uuid == "model_training":
            self.log.warning("Cannot delete model training data!")
            return

        if self.endpoint_inference_path:
            full_path = f"{self.endpoint_inference_path}/{inference_run_uuid}"
            # Check if there are any objects at the path
            if wr.s3.list_objects(full_path):
                wr.s3.delete_objects(path=full_path)
                self.log.important(f"Deleted inference run {inference_run_uuid} for {self.model_name}")
            else:
                self.log.warning(f"Inference run {inference_run_uuid} not found for {self.model_name}!")
        else:
            self.log.warning(f"No inference data found for {self.model_name}!")

    def get_inference_metrics(self, capture_uuid: str = "latest") -> Union[pd.DataFrame, None]:
        """Retrieve the inference performance metrics for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid or "training" (default: "latest")
        Returns:
            pd.DataFrame: DataFrame of the Model Metrics

        Note:
            If a capture_uuid isn't specified this will try to return something reasonable
        """
        # Try to get the auto_capture 'training_holdout' or the training
        if capture_uuid == "latest":
            metrics_df = self.get_inference_metrics("auto_inference")
            return metrics_df if metrics_df is not None else self.get_inference_metrics("model_training")

        # Grab the metrics captured during model training (could return None)
        if capture_uuid == "model_training":
            # Sanity check the sageworks metadata
            if self.sageworks_meta() is None:
                error_msg = f"Model {self.model_name} has no sageworks_meta(). Either onboard() or delete this model!"
                self.log.critical(error_msg)
                raise ValueError(error_msg)

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

        # Sanity check the sageworks metadata
        if self.sageworks_meta() is None:
            error_msg = f"Model {self.model_name} has no sageworks_meta(). Either onboard() or delete this model!"
            self.log.critical(error_msg)
            raise ValueError(error_msg)

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
        return self.model_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for the Model Package Group"""
        return self.group_arn()

    def group_arn(self) -> Union[str, None]:
        """AWS ARN (Amazon Resource Name) for the Model Package Group"""
        return self.model_meta["ModelPackageGroupArn"] if self.model_meta else None

    def model_package_arn(self) -> Union[str, None]:
        """AWS ARN (Amazon Resource Name) for the Latest Model Package (within the Group)"""
        if self.latest_model is None:
            return None
        return self.latest_model["ModelPackageArn"]

    def container_info(self) -> Union[dict, None]:
        """Container Info for the Latest Model Package"""
        return self.latest_model["InferenceSpecification"]["Containers"][0] if self.latest_model else None

    def container_image(self) -> str:
        """Container Image for the Latest Model Package"""
        return self.container_info()["Image"]

    def aws_url(self):
        """The AWS URL for looking at/querying this model"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        if self.latest_model is None:
            return "-"
        return self.latest_model["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        if self.latest_model is None:
            return "-"
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

        # Remove any health tags
        self.remove_health_tag("no_endpoint")

        # A new endpoint means we need to refresh our inference path
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.endpoint_inference_path = self.get_endpoint_inference_path()

    def remove_endpoint(self, endpoint_name: str):
        """Remove this endpoint from the set of registered endpoints for the model

        Args:
            endpoint_name (str): Name of the endpoint
        """
        self.log.important(f"Removing Endpoint {endpoint_name} from Model {self.uuid}...")
        registered_endpoints = set(self.sageworks_meta().get("sageworks_registered_endpoints", []))
        registered_endpoints.discard(endpoint_name)
        self.upsert_sageworks_meta({"sageworks_registered_endpoints": list(registered_endpoints)})

        # If we have NO endpionts, then set a health tags
        if not registered_endpoints:
            self.add_health_tag("no_endpoint")
            self.details(recompute=True)

        # A new endpoint means we need to refresh our inference path
        time.sleep(2)

    def endpoints(self) -> list[str]:
        """Get the list of registered endpoints for this Model

        Returns:
            list[str]: List of registered endpoints
        """
        return self.sageworks_meta().get("sageworks_registered_endpoints", [])

    def get_endpoint_inference_path(self) -> Union[str, None]:
        """Get the S3 Path for the Inference Data

        Returns:
            str: S3 Path for the Inference Data (or None if not found)
        """

        # Look for any Registered Endpoints
        registered_endpoints = self.sageworks_meta().get("sageworks_registered_endpoints")

        # Note: We may have 0 to N endpoints, so we find the one with the most recent artifacts
        if registered_endpoints:
            endpoint_inference_base = self.endpoints_s3_path + "/inference/"
            endpoint_inference_paths = [endpoint_inference_base + e for e in registered_endpoints]
            inference_path = newest_path(endpoint_inference_paths, self.sm_session)
            if inference_path is None:
                self.log.important(f"No inference data found for {self.model_name}!")
                self.log.important(f"Returning default inference path for {registered_endpoints[0]}...")
                self.log.important(f"{endpoint_inference_paths[0]}")
                return endpoint_inference_paths[0]
            else:
                return inference_path
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
        self.log.info("Computing Model Details...")
        details = self.summary()
        details["pipeline"] = self.get_pipeline()
        details["model_type"] = self.model_type.value
        details["model_package_group_arn"] = self.group_arn()
        details["model_package_arn"] = self.model_package_arn()

        # Sanity check is we have models in the group
        if self.latest_model is None:
            self.log.warning(f"Model Package Group {self.model_name} has no models!")
            return details

        # Grab the Model Details
        details["description"] = self.latest_model.get("ModelPackageDescription", "-")
        details["version"] = self.latest_model["ModelPackageVersion"]
        details["status"] = self.latest_model["ModelPackageStatus"]
        details["approval_status"] = self.latest_model.get("ModelApprovalStatus", "unknown")
        details["image"] = self.container_image().split("/")[-1]  # Shorten the image uri

        # Grab the inference and container info
        inference_spec = self.latest_model["InferenceSpecification"]
        container_info = self.container_info()
        details["framework"] = container_info.get("Framework", "unknown")
        details["framework_version"] = container_info.get("FrameworkVersion", "unknown")
        details["inference_types"] = inference_spec["SupportedRealtimeInferenceInstanceTypes"]
        details["transform_types"] = inference_spec["SupportedTransformInstanceTypes"]
        details["content_types"] = inference_spec["SupportedContentTypes"]
        details["response_types"] = inference_spec["SupportedResponseMIMETypes"]
        details["model_metrics"] = self.get_inference_metrics()
        if self.model_type == ModelType.CLASSIFIER:
            details["confusion_matrix"] = self.confusion_matrix()
            details["predictions"] = None
        elif self.model_type in [ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]:
            details["confusion_matrix"] = None
            details["predictions"] = self.get_inference_predictions()
        else:
            details["confusion_matrix"] = None
            details["predictions"] = None

        # Grab the inference metadata
        details["inference_meta"] = self.get_inference_metadata()

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
        model_type = input("Model Type? (classifier, regressor, quantile_regressor, unsupervised, transformer): ")
        if model_type == "classifier":
            self._set_model_type(ModelType.CLASSIFIER)
        elif model_type == "regressor":
            self._set_model_type(ModelType.REGRESSOR)
        elif model_type == "quantile_regressor":
            self._set_model_type(ModelType.QUANTILE_REGRESSOR)
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

        # Is our input data set?
        if self.get_input() in ["", "unknown"] or ask_everything:
            input_data = input("Input Data?: ")
            if input_data not in ["None", "none", "", "unknown"]:
                self.set_input(input_data)

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

        # Model Class Labels (if it's a classifier)
        if self.model_type == ModelType.CLASSIFIER:
            class_labels = self.class_labels()
            if class_labels is None or ask_everything:
                class_labels = input("Class Labels? (use commas): ")
                class_labels = [e.strip() for e in class_labels.split(",")]
                if class_labels in [["None"], ["none"], [""]]:
                    class_labels = None
            self.set_class_labels(class_labels)

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
        self.set_status("ready")

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details(recompute=True)
        return True

    def delete(self):
        """Delete the Model Packages and the Model Group"""
        if not self.exists():
            self.log.warning(f"Trying to delete an Model that doesn't exist: {self.uuid}")

        # Call the Class Method to delete the Model Group
        ModelCore.managed_delete(model_group_name=self.uuid)

    @classmethod
    def managed_delete(cls, model_group_name: str):
        """Delete the Model Packages, Model Group, and S3 Storage Objects

        Args:
            model_group_name (str): The name of the Model Group to delete
        """
        # Check if the model group exists in SageMaker
        try:
            cls.sm_client.describe_model_package_group(ModelPackageGroupName=model_group_name)
        except ClientError as e:
            if e.response["Error"]["Code"] in ["ValidationException", "ResourceNotFound"]:
                cls.log.info(f"Model Group {model_group_name} not found!")
                return
            else:
                raise  # Re-raise unexpected errors

        # Delete Model Packages within the Model Group
        try:
            paginator = cls.sm_client.get_paginator("list_model_packages")
            for page in paginator.paginate(ModelPackageGroupName=model_group_name):
                for model_package in page["ModelPackageSummaryList"]:
                    package_arn = model_package["ModelPackageArn"]
                    cls.log.info(f"Deleting Model Package {package_arn}...")
                    cls.sm_client.delete_model_package(ModelPackageName=package_arn)
        except ClientError as e:
            cls.log.error(f"Error while deleting model packages: {e}")
            raise

        # Delete the Model Package Group
        cls.log.info(f"Deleting Model Group {model_group_name}...")
        cls.sm_client.delete_model_package_group(ModelPackageGroupName=model_group_name)

        # Delete S3 training artifacts
        s3_delete_path = f"{cls.models_s3_path}/training/{model_group_name}/"
        cls.log.info(f"Deleting S3 Objects at {s3_delete_path}...")
        wr.s3.delete_objects(s3_delete_path, boto3_session=cls.boto3_session)

        # Delete any dataframes that were stored in the Dataframe Cache
        cls.log.info("Deleting Dataframe Cache...")
        cls.df_cache.delete_recursive(model_group_name)

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
        try:
            return ModelType(model_type)
        except ValueError:
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
                self.log.important(f"No training job metrics found for {self.training_job_name}")
                self.upsert_sageworks_meta({"sageworks_training_metrics": None, "sageworks_training_cm": None})
                return
            if self.model_type in [ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]:
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
            self.log.important(f"No training job metrics found for {self.training_job_name}")
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

    def _load_inference_metrics(self, capture_uuid: str = "auto_inference"):
        """Internal: Retrieve the inference model metrics for this model
                     and load the data into the SageWorks Metadata

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "auto_inference")
        Notes:
            This may or may not exist based on whether an Endpoint ran Inference
        """
        s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_metrics.csv"
        inference_metrics = pull_s3_data(s3_path)

        # Store data into the SageWorks Metadata
        metrics_storage = None if inference_metrics is None else inference_metrics.to_dict("records")
        self.upsert_sageworks_meta({"sageworks_inference_metrics": metrics_storage})

    def _load_inference_cm(self, capture_uuid: str = "auto_inference"):
        """Internal: Pull the inference Confusion Matrix for this model
                     and load the data into the SageWorks Metadata

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "auto_inference")

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

    def get_inference_metadata(self, capture_uuid: str = "auto_inference") -> Union[pd.DataFrame, None]:
        """Retrieve the inference metadata for this model

        Args:
            capture_uuid (str, optional): A specific capture_uuid (default: "auto_inference")

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

    def get_inference_predictions(self, capture_uuid: str = "auto_inference") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: training_holdout)

        Returns:
            pd.DataFrame: DataFrame of the Captured Predictions (might be None)
        """
        self.log.important(f"Grabbing {capture_uuid} predictions for {self.model_name}...")

        # Sanity check that the model should have predictions
        has_predictions = self.model_type in [ModelType.CLASSIFIER, ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]
        if not has_predictions:
            self.log.warning(f"No Predictions for {self.model_name}...")
            return None

        # Special case for model_training
        if capture_uuid == "model_training":
            return self._get_validation_predictions()

        # Construct the S3 path for the Inference Predictions
        s3_path = f"{self.endpoint_inference_path}/{capture_uuid}/inference_predictions.csv"
        return pull_s3_data(s3_path)

    def _get_validation_predictions(self) -> Union[pd.DataFrame, None]:
        """Internal: Retrieve the captured prediction results for this model

        Returns:
            pd.DataFrame: DataFrame of the Captured Validation Predictions (might be None)
        """
        # Sanity check the training path (which may or may not exist)
        if self.model_training_path is None:
            self.log.warning(f"No Validation Predictions for {self.model_name}...")
            return None
        self.log.important(f"Grabbing Validation Predictions for {self.model_name}...")
        s3_path = f"{self.model_training_path}/validation_predictions.csv"
        df = pull_s3_data(s3_path)
        return df

    def _extract_training_job_name(self) -> Union[str, None]:
        """Internal: Extract the training job name from the ModelDataUrl"""
        try:
            model_data_url = self.container_info()["ModelDataUrl"]
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

    def shapley_values(self, capture_uuid: str = "auto_inference") -> Union[list[pd.DataFrame], pd.DataFrame, None]:
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
        if self.model_type in [ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]:
            s3_path = f"{shapley_s3_path}/inference_shap_values.csv"
            return pull_s3_data(s3_path)


if __name__ == "__main__":
    """Exercise the ModelCore Class"""
    from pprint import pprint

    # Grab a ModelCore object and pull some information from it
    my_model = ModelCore("abalone-regression")

    # Call the various methods

    # Let's do a check/validation of the Model
    print(f"Model Check: {my_model.exists()}")

    # Make sure the model is 'ready'
    my_model.onboard()

    # Get the ARN of the Model Group
    print(f"Model Group ARN: {my_model.group_arn()}")
    print(f"Model Package ARN: {my_model.arn()}")

    # Get the container info
    print("Container Info:")
    pprint(my_model.container_info())

    # Get the tags associated with this Model
    print(f"Tags: {my_model.get_tags()}")

    # Get the SageWorks metadata associated with this Model
    print("SageWorks Meta:")
    pprint(my_model.sageworks_meta())

    # Get creation time
    print(f"Created: {my_model.created()}")

    # Get training job name
    print(f"Training Job: {my_model.training_job_name}")

    # List any inference runs
    print(f"Inference Runs: {my_model.list_inference_runs()}")

    # Get any captured metrics from the training job
    print("Model Metrics:")
    print(my_model.get_inference_metrics())

    print("Confusion Matrix: (might be None)")
    print(my_model.confusion_matrix())

    # Grab our regression predictions from S3
    print("Captured Predictions: (might be None)")
    print(my_model.get_inference_predictions())

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

    # Delete the Model
    # ModelCore.managed_delete("wine-classification")

    # Check for a model that doesn't exist
    my_model = ModelCore("empty-model-group")
