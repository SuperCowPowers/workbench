"""Model: SageWorks Model Class"""
from datetime import datetime
import urllib.parse
from typing import Union

import pandas as pd
from sagemaker import TrainingJobAnalytics

# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


class Model(Artifact):
    """Model: SageWorks Model Class

    Common Usage:
        my_model = Model(model_uuid)
        my_model.summary()
        my_model.details()
    """

    def __init__(self, model_uuid: str, force_refresh: bool = False):
        """Model Initialization
        Args:
            model_uuid (str): Name of Model in SageWorks.
            force_refresh (bool, optional): Force a refresh of the AWS Broker. Defaults to False.
        """
        # Call SuperClass Initialization
        super().__init__(model_uuid)

        # Grab an AWS Metadata Broker object and pull information for Models
        self.model_name = model_uuid
        aws_meta = self.aws_broker.get_metadata(ServiceCategory.MODELS, force_refresh=force_refresh)
        self.model_meta = aws_meta.get(self.model_name)
        if self.model_meta is None:
            self.log.warning(f"Could not find model {self.model_name} within current visibility scope")
            self.latest_model = None
        else:
            self.latest_model = self.model_meta[0]
            self.description = self.latest_model["ModelPackageDescription"]
            self.training_job_name = self._extract_training_job_name()
            self._model_metrics, self._confusion_matrix = self._training_job_metrics()

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

    def model_type(self) -> str:
        """Return the model type (classifier or regressor)"""
        if "classification" in self.sageworks_tags():
            return "classifier"
        elif "regression" in self.sageworks_tags():
            return "regressor"
        else:
            return "unknown"

    def model_metrics(self) -> pd.DataFrame:
        """Retrieve the training metrics for this model
        Returns:
            pd.DataFrame: DataFrame of the Model Metrics
        """
        return self._model_metrics

    def confusion_matrix(self) -> Union[pd.DataFrame, None]:
        """Retrieve the confusion_matrix for this model
        Returns:
            pd.DataFrame: DataFrame of the Confusion Matrix (might be None)
        """
        return self._confusion_matrix

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

    def model_arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for the Model Package Group"""
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

    def details(self) -> dict:
        """Additional Details about this Endpoint"""
        details = self.summary()
        details["model_package_group_arn"] = self.group_arn()
        details["model_package_arn"] = self.model_arn()
        aws_meta = self.aws_meta()
        details["description"] = aws_meta["ModelPackageDescription"]
        details["status"] = aws_meta["ModelPackageStatus"]
        details["approval_status"] = aws_meta["ModelApprovalStatus"]
        package_details = aws_meta["ModelPackageDetails"]
        inference_spec = package_details["InferenceSpecification"]
        container = inference_spec["Containers"][0]
        image_short = container["Image"].split("/")[-1]
        details["image"] = image_short
        details["framework"] = container["Framework"]
        details["framework_version"] = container["FrameworkVersion"]
        details["inference_types"] = inference_spec["SupportedRealtimeInferenceInstanceTypes"]
        details["transform_types"] = inference_spec["SupportedTransformInstanceTypes"]
        details["content_types"] = inference_spec["SupportedContentTypes"]
        details["response_types"] = inference_spec["SupportedResponseMIMETypes"]
        details["model_metrics"] = self.model_metrics()
        details["confusion_matrix"] = self.confusion_matrix()
        return details

    def make_ready(self) -> bool:
        """This is a BLOCKING method that will wait until the Model is ready"""
        self.details()
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

        # Now delete the Model Package Group
        self.log.info(f"Deleting Model Group {self.model_name}...")
        self.sm_client.delete_model_package_group(ModelPackageGroupName=self.model_name)

    def _training_job_metrics(self) -> (pd.DataFrame, pd.DataFrame):
        """Internal: Grab any captured metrics from the training job for this model"""
        try:
            df = TrainingJobAnalytics(training_job_name=self.training_job_name).dataframe()
            if self.model_type() == "regressor":
                if "timestamp" in df.columns:
                    df = df.drop(columns=["timestamp"])
                return df, None
        except KeyError:
            self.log.warning(f"No training job metrics found for {self.training_job_name}")
            return None, None

        # We need additional processing for classification metrics
        if self.model_type() == "classifier":
            metrics_df, cm_df = self._process_classification_metrics(df)
            return metrics_df, cm_df

    def _extract_training_job_name(self) -> str:
        """Internal: Extract the training job name from the ModelDataUrl"""
        model_data_url = self.latest_model["ModelPackageDetails"]["InferenceSpecification"]["Containers"][0][
            "ModelDataUrl"
        ]
        parsed_url = urllib.parse.urlparse(model_data_url)
        training_job_name = parsed_url.path.lstrip("/").split("/")[0]
        return training_job_name

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
    my_model = Model("abalone-regression", force_refresh=True)

    # Call the various methods

    # Let's do a check/validation of the Model
    print(f"Model Check: {my_model.exists()}")

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

    # Delete the Model
    # my_model.delete()
