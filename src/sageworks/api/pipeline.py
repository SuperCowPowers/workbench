"""Pipeline: Manages the details around a SageWorks Pipeline, including Execution"""

import sys
import logging
import json
import awswrangler as wr
from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from sageworks.core.pipelines.pipeline_executor import PipelineExecutor


class Pipeline:
    """Pipeline: SageWorks Pipeline API Class

    Common Usage:
        ```python
        my_pipeline = Pipeline("name")
        my_pipeline.details()
        my_pipeline.execute()  # Execute entire pipeline
        my_pipeline.execute_partial(["data_source", "feature_set"])
        my_pipeline.execute_partial(["model", "endpoint"])
        ```
    """

    def __init__(self, name: str):
        """Pipeline Init Method"""
        self.log = logging.getLogger("sageworks")
        self.name = name

        # Grab our SageWorks Bucket from Config
        self.cm = ConfigManager()
        self.sageworks_bucket = self.cm.get_config("SAGEWORKS_BUCKET")
        if self.sageworks_bucket is None:
            self.log = logging.getLogger("sageworks")
            self.log.critical("Could not find ENV var for SAGEWORKS_BUCKET!")
            sys.exit(1)

        # Set the S3 Path for this Pipeline
        self.bucket = self.sageworks_bucket
        self.key = f"pipelines/{self.name}.json"
        self.s3_path = f"s3://{self.bucket}/{self.key}"

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session
        self.s3_client = self.boto3_session.client("s3")

        # If this S3 Path exists, load the Pipeline
        if wr.s3.does_object_exist(self.s3_path):
            self.pipeline = self._get_pipeline()
        else:
            self.log.warning(f"Pipeline {self.name} not found at {self.s3_path}")
            self.pipeline = None

    def set_input(self, input: Union[str, pd.DataFrame], artifact: str = "data_source"):
        """Set the input for the Pipeline

        Args:
            input (Union[str, pd.DataFrame]): The input for the Pipeline
            artifact (str): The artifact to set the input for (default: "data_source")
        """
        self.pipeline[artifact]["input"] = input

    def set_training_holdouts(self, id_column: str, holdout_ids: list[str]):
        """Set the input for the Pipeline

        Args:
            id_column (str): The column name of the unique identifier
            holdout_ids (list[str]): The list of unique identifiers to hold out
        """
        self.pipeline["feature_set"]["id_column"] = id_column
        self.pipeline["feature_set"]["holdout_ids"] = holdout_ids

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

    def report_settable_fields(self, pipeline: dict = {}, path: str = "") -> None:
        """
        Recursively finds and prints keys with settable fields in a JSON-like dictionary.

        Args:
        pipeline (dict): pipeline (or sub pipeline) to process.
        path (str): Current path to the key, used for nested dictionaries.
        """
        # Grab the entire pipeline if not provided (first call)
        if not pipeline:
            self.log.important(f"Checking Pipeline: {self.name}...")
            pipeline = self.pipeline
        for key, value in pipeline.items():
            if isinstance(value, dict):
                # Recurse into sub-dictionary
                self.report_settable_fields(value, path + key + " -> ")
            elif isinstance(value, str) and value.startswith("<<") and value.endswith(">>"):
                # Check if required or optional
                required = "[Required]" if "required" in value else "[Optional]"
                self.log.important(f"{required} Path: {path + key}")

    def delete(self):
        """Pipeline Deletion"""
        self.log.info(f"Deleting Pipeline: {self.name}...")
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

    log = logging.getLogger("sageworks")

    # Temp testing
    my_pipeline = Pipeline("aqsol_pipeline_v1")
    my_pipeline.set_input("s3://sageworks-public-data/comp_chem/aqsol_public_data.csv")
    my_pipeline.execute_partial(["model", "endpoint"])
    exit(0)

    # Retrieve an existing Pipeline
    my_pipeline = Pipeline("abalone_pipeline_v1")
    print(my_pipeline)

    # Report on any settable fields in the pipeline
    my_pipeline.report_settable_fields()

    # Retrieve an existing Pipeline
    my_pipeline = Pipeline("abalone_pipeline_v2")

    # Report on any settable fields in the pipeline
    my_pipeline.report_settable_fields()

    # Try running a pipeline without a required field set
    # Assert that a RuntimeError is raised
    try:
        my_pipeline.execute()
        assert False, "Expected a RuntimeError to be raised!"
    except RuntimeError:
        log.info("Expected Exection = AOK :)")

    # Set the input for the Pipeline
    my_pipeline.set_input("s3://sageworks-public-data/common/abalone.csv")

    # Set the hold out ids for the Pipeline
    my_pipeline.set_training_holdouts("id", list(range(100)))

    # Now we can execute the pipeline
    my_pipeline.execute_partial(["data_source", "feature_set"])

    # ds = DataSource("solubility_featurized_ds")
    # df = ds.pull_dataframe()
    # my_pipeline.set_input(df)

    # Execute the Pipeline
    # my_pipeline.execute()
    # my_pipeline.execute_partial(["data_source", "feature_set"])
    # my_pipeline.execute_partial(["model", "endpoint"])
