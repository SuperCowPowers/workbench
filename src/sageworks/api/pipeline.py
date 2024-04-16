"""Pipeline: Manages the details around a SageWorks Pipeline, including Execution"""

import sys
import logging
import json
import awswrangler as wr

# SageWorks Imports
from sageworks.utils.sageworks_cache import SageWorksCache
from sageworks.utils.config_manager import ConfigManager
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

# from sageworks.api import DataSource, FeatureSet, Model, Endpoint


class Pipeline:
    """Pipeline: SageWorks Pipeline API Class

    Common Usage:
        ```
        my_pipeline = Pipeline("pipeline_name")
        my_pipeline.details()
        my_pipeline.execute()  # Execute entire pipeline
        my_pipeline.execute_partial(["data_source", "feature_set"])
        my_pipeline.execute_partial(["model", "endpoint"])
        ```
    """

    def __init__(self, name: str):
        """Pipeline Init Method"""
        self.log = logging.getLogger("sageworks")
        self.pipeline_name = name

        # Grab our SageWorks Bucket from Config
        self.cm = ConfigManager()
        self.sageworks_bucket = self.cm.get_config("SAGEWORKS_BUCKET")
        if self.sageworks_bucket is None:
            self.log = logging.getLogger("sageworks")
            self.log.critical("Could not find ENV var for SAGEWORKS_BUCKET!")
            sys.exit(1)

        # Set the S3 Path for this Pipeline
        self.bucket = self.sageworks_bucket
        self.key = f"pipelines/{self.pipeline_name}.json"
        self.s3_path = f"s3://{self.bucket}/{self.key}"

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto_session = AWSAccountClamp().boto_session()
        self.s3_client = self.boto_session.client("s3")

        # If this S3 Path exists, load the Pipeline
        if wr.s3.does_object_exist(self.s3_path):
            self.pipeline = self._get_pipeline()
        else:
            self.log.warning(f"Pipeline {self.pipeline_name} not found at {self.s3_path}")
            self.pipeline = None

        # Data Storage Cache
        self.data_storage = SageWorksCache(prefix="data_storage")

    def details(self, recompute=False) -> dict:
        """Pipeline Details

        Args:
            recompute (bool, optional): Recompute the details (default: False)

        Returns:
            dict: A dictionary of details about the Pipeline
        """
        # Check if we have cached version of the Pipeline Details
        storage_key = f"pipeline:{self.pipeline_name}:details"
        cached_details = self.data_storage.get(storage_key)
        if cached_details and not recompute:
            return cached_details

        self.log.important("Recomputing Pipeline Details...")
        details = {}
        details["name"] = self.pipeline_name
        details["s3_path"] = self.s3_path
        details["pipeline"] = self.pipeline

        # Cache the details
        self.data_storage.set(storage_key, details)

        # Return the details
        return details

    def execute(self):
        """Execute the entire Pipeline

        Raises:
            RunTimeException: If the pipeline execution fails in any way
        """
        self.log.important(f"Executing Pipeline: {self.pipeline_name}...")

    def delete(self):
        """Pipeline Deletion"""
        self.log.info(f"Deleting Pipeline: {self.pipeline_name}...")
        self.data_storage.delete(f"pipeline:{self.pipeline_name}:details")
        wr.s3.delete_objects(self.s3_path)

    def _get_pipeline(self) -> dict:
        """Internal: Get the pipeline as a JSON object from the specified S3 bucket and key."""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
        json_object = json.loads(response["Body"].read())
        return json_object

    def __repr__(self) -> str:
        """String representation of this pipeline

        Returns:
            str: String representation of this pipeline
        """
        # Class name and details
        class_name = self.__class__.__name__
        details = json.dumps(self.details(), indent=4)
        return f"{class_name}({details})"


if __name__ == "__main__":
    """Exercise the Pipeline Class"""
    from pprint import pprint

    # Retrieve an existing Pipeline
    my_pipeline = Pipeline("abalone_pipeline_v1")
    pprint(my_pipeline.details(recompute=True))

    # Execute the Pipeline
    my_pipeline.execute()

    # Print the Representation of the Pipeline
    print(my_pipeline)
