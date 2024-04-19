"""Pipeline: Manages the details around a SageWorks Pipeline, including Execution"""

import sys
import logging
import json
import awswrangler as wr
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.utils.sageworks_cache import SageWorksCache
from sageworks.utils.config_manager import ConfigManager
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.core.pipelines.pipeline_executor import PipelineExecutor


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

    def set_input(self, input: Union[str, pd.DataFrame]):
        """Set the input for the Pipeline

        Args:
            input (Union[str, pd.DataFrame]): The input for the Pipeline
        """
        self.pipeline["data_source"]["input"] = input

    def execute(self):
        """Execute the entire Pipeline

        Raises:
            RunTimeException: If the pipeline execution fails in any way
        """
        pipeline_executor = PipelineExecutor(self)
        pipeline_executor.execute()

    def execute_partial(self, subset: list):
        """Execute a partial Pipeline

        Args:
            subset (list): A subset of the pipeline to execute

        Raises:
            RunTimeException: If the pipeline execution fails in any way
        """
        pipeline_executor = PipelineExecutor(self)
        pipeline_executor.execute_partial(subset)

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
        pipeline_details = json.dumps(self.pipeline, indent=4)
        return f"{class_name}({pipeline_details})"


if __name__ == "__main__":
    """Exercise the Pipeline Class"""
    from sageworks.api import DataSource

    # Retrieve an existing Pipeline
    my_pipeline = Pipeline("test_solubility_class_nightly_80_v0")
    print(my_pipeline)

    # Set the input for the Pipeline (this is just for testing)
    ds = DataSource("solubility_featurized_ds")
    df = ds.pull_dataframe()
    my_pipeline.set_input(df)

    # Execute the Pipeline
    # my_pipeline.execute()
    # my_pipeline.execute_partial(["data_source", "feature_set"])
    my_pipeline.execute_partial(["model", "endpoint"])
