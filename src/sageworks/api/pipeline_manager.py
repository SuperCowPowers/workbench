"""PipelineManager: Manages SageWorks Pipelines, listing, creating, and saving them."""

import sys
import logging
import json

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from sageworks.api import DataSource, FeatureSet, Model, Endpoint


class PipelineManager:
    """PipelineManager: Manages SageWorks Pipelines, listing, creating, and saving them.

    Common Usage:
        ```python
        my_manager = PipelineManager()
        my_manager.list_pipelines()
        abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")
        my_manager.save_pipeline("abalone_pipeline_v1", abalone_pipeline)
        ```
    """

    def __init__(self):
        """Pipeline Init Method"""
        self.log = logging.getLogger("sageworks")

        # Grab our SageWorks Bucket from Config
        self.cm = ConfigManager()
        self.sageworks_bucket = self.cm.get_config("SAGEWORKS_BUCKET")
        if self.sageworks_bucket is None:
            self.log = logging.getLogger("sageworks")
            self.log.critical("Could not find ENV var for SAGEWORKS_BUCKET!")
            sys.exit(1)

        # Set the S3 Path for Pipelines
        self.bucket = self.sageworks_bucket
        self.prefix = "pipelines/"
        self.pipelines_s3_path = f"s3://{self.sageworks_bucket}/pipelines/"

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session

        # Read all the Pipelines from this S3 path
        self.s3_client = self.boto3_session.client("s3")

    def list_pipelines(self) -> list:
        """List all the Pipelines in the S3 Bucket

        Returns:
            list: A list of Pipeline names and details
        """
        # List objects using the S3 client
        response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)

        # Check if there are objects
        if "Contents" in response:
            # Process the list of dictionaries (we only need the filename, the LastModified, and the Size)
            pipelines = [
                {
                    "name": pipeline["Key"].split("/")[-1].replace(".json", ""),
                    "last_modified": pipeline["LastModified"],
                    "size": pipeline["Size"],
                }
                for pipeline in response["Contents"]
            ]
            return pipelines
        else:
            self.log.important(f"No pipelines found at {self.pipelines_s3_path}...")
            return []

    # Create a new Pipeline from an Endpoint
    def create_from_endpoint(self, endpoint_name: str) -> dict:
        """Create a Pipeline from an Endpoint

        Args:
            endpoint_name (str): The name of the Endpoint

        Returns:
            dict: A dictionary of the Pipeline
        """
        self.log.important(f"Creating Pipeline from Endpoint: {endpoint_name}...")
        pipeline = {}
        endpoint = Endpoint(endpoint_name)
        model = Model(endpoint.get_input())
        feature_set = FeatureSet(model.get_input())
        data_source = DataSource(feature_set.get_input())
        s3_source = data_source.get_input()
        for name in ["data_source", "feature_set", "model", "endpoint"]:
            artifact = locals()[name]
            pipeline[name] = {"name": artifact.uuid, "tags": artifact.get_tags(), "input": artifact.get_input()}
            if name == "model":
                pipeline[name]["model_type"] = artifact.model_type.value
                pipeline[name]["target_column"] = artifact.target()
                pipeline[name]["feature_list"] = artifact.features()

        # Return the Pipeline
        return pipeline

    # Publish a Pipeline to SageWorks
    def publish_pipeline(self, name: str, pipeline: dict):
        """Save a Pipeline to S3

        Args:
            name (str): The name of the Pipeline
            pipeline (dict): The Pipeline to save
        """
        key = f"{self.prefix}{name}.json"
        self.log.important(f"Saving {name} to S3: {self.bucket}/{key}...")

        # Save the pipeline as an S3 JSON object
        self.s3_client.put_object(Body=json.dumps(pipeline, indent=4), Bucket=self.bucket, Key=key)

    def delete_pipeline(self, name: str):
        """Delete a Pipeline from S3

        Args:
            name (str): The name of the Pipeline to delete
        """
        key = f"{self.prefix}{name}.json"
        self.log.important(f"Deleting {name} from S3: {self.bucket}/{key}...")

        # Delete the pipeline object from S3
        self.s3_client.delete_object(Bucket=self.bucket, Key=key)

    # Save a Pipeline to a local file
    def save_pipeline_to_file(self, pipeline: dict, filepath: str):
        """Save a Pipeline to a local file

        Args:
            pipeline (dict): The Pipeline to save
            filepath (str): The path to save the Pipeline
        """

        # Sanity check the filepath
        if not filepath.endswith(".json"):
            filepath += ".json"

        # Save the pipeline as a local JSON file
        with open(filepath, "w") as fp:
            json.dump(pipeline, fp, indent=4)

    def load_pipeline_from_file(self, filepath: str) -> dict:
        """Load a Pipeline from a local file

        Args:
            filepath (str): The path of the Pipeline to load

        Returns:
            dict: The Pipeline loaded from the file
        """

        # Load a pipeline as a local JSON file
        with open(filepath, "r") as fp:
            pipeline = json.load(fp)
            return pipeline

    def publish_pipeline_from_file(self, filepath: str):
        """Publish a Pipeline to SageWorks from a local file

        Args:
            filepath (str): The path of the Pipeline to publish
        """

        # Load a pipeline as a local JSON file
        pipeline = self.load_pipeline_from_file(filepath)

        # Get the pipeline name
        pipeline_name = filepath.split("/")[-1].replace(".json", "")

        # Publish the Pipeline
        self.publish_pipeline(pipeline_name, pipeline)


if __name__ == "__main__":
    """Exercise the Pipeline Class"""
    from pprint import pprint

    # Create a PipelineManager
    my_manager = PipelineManager()

    # List the Pipelines
    print("Listing Pipelines...")
    pprint(my_manager.list_pipelines())

    # Create a Pipeline from an Endpoint
    abalone_pipeline = my_manager.create_from_endpoint("abalone-regression-end")

    # Publish the Pipeline
    my_manager.publish_pipeline("abalone_pipeline_v1", abalone_pipeline)
