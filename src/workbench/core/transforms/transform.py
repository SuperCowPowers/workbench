"""Transform: Base Class for all transforms within Workbench
Inherited Classes must implement the abstract transform_impl() method"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Union, final
import logging

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager, FatalConfigError


class TransformInput(Enum):
    """Enumerated Types for Workbench Transform Inputs"""

    LOCAL_FILE = auto()
    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()


class TransformOutput(Enum):
    """Enumerated Types for Workbench Transform Outputs"""

    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()
    ENDPOINT = auto()


class Transform(ABC):
    """Transform: Abstract Base Class for all transforms within Workbench. Inherited Classes
    must implement the abstract transform_impl() method"""

    def __init__(self, input_uuid: str, output_uuid: str, catalog_db: str = "workbench"):
        """Transform Initialization

        Args:
            input_uuid (str): The UUID of the Input Artifact
            output_uuid (str): The UUID of the Output Artifact
            catalog_db (str): The AWS Data Catalog Database to use (default: "workbench")
        """

        self.log = logging.getLogger("workbench")
        self.input_type = None
        self.output_type = None
        self.output_tags = ""
        self.input_uuid = str(input_uuid)  # Occasionally we get a pathlib.Path object
        self.output_uuid = str(output_uuid)  # Occasionally we get a pathlib.Path object
        self.output_meta = {"workbench_input": self.input_uuid}
        self.data_catalog_db = catalog_db

        # Grab our Workbench Bucket
        cm = ConfigManager()
        if not cm.config_okay():
            self.log.error("Workbench Configuration Incomplete...")
            self.log.error("Run the 'workbench' command and follow the prompts...")
            raise FatalConfigError()
        self.workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
        self.data_sources_s3_path = "s3://" + self.workbench_bucket + "/data-sources"
        self.feature_sets_s3_path = "s3://" + self.workbench_bucket + "/feature-sets"
        self.models_s3_path = "s3://" + self.workbench_bucket + "/models"
        self.endpoints_sets_s3_path = "s3://" + self.workbench_bucket + "/endpoints"

        # Grab a Workbench Role ARN, Boto3, SageMaker Session, and SageMaker Client
        self.aws_account_clamp = AWSAccountClamp()
        self.workbench_role_arn = self.aws_account_clamp.aws_session.get_workbench_execution_role_arn()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.sm_session = self.aws_account_clamp.sagemaker_session()
        self.sm_client = self.aws_account_clamp.sagemaker_client()

        # Delimiter for storing lists in AWS Tags
        self.tag_delimiter = "::"

    @abstractmethod
    def transform_impl(self, **kwargs):
        """Abstract Method: Implement the Transformation from Input to Output"""
        pass

    def pre_transform(self, **kwargs):
        """Perform any Pre-Transform operations"""
        self.log.debug("Pre-Transform...")

    @abstractmethod
    def post_transform(self, **kwargs):
        """Post-Transform ensures that the output Artifact is ready for use"""
        pass

    def set_output_tags(self, tags: Union[list, str]):
        """Set the tags that will be associated with the output object
        Args:
            tags (Union[list, str]): The list of tags or a '::' separated string of tags"""
        if isinstance(tags, list):
            self.output_tags = self.tag_delimiter.join(tags)
        else:
            self.output_tags = tags

    def add_output_meta(self, meta: dict):
        """Add additional metadata that will be associated with the output artifact
        Args:
            meta (dict): A dictionary of metadata"""
        self.output_meta = self.output_meta | meta

    @staticmethod
    def convert_to_aws_tags(metadata: dict):
        """Convert a dictionary to the AWS tag format (list of dicts)
        [ {Key: key_name, Value: value}, {..}, ...]"""
        return [{"Key": key, "Value": value} for key, value in metadata.items()]

    def get_aws_tags(self):
        """Get the metadata/tags and convert them into AWS Tag Format"""
        # Set up our metadata storage
        workbench_meta = {"workbench_tags": self.output_tags}
        for key, value in self.output_meta.items():
            workbench_meta[key] = value
        aws_tags = self.convert_to_aws_tags(workbench_meta)
        return aws_tags

    @final
    def transform(self, **kwargs):
        """Perform the Transformation from Input to Output with pre_transform() and post_transform() invocations"""
        self.pre_transform(**kwargs)
        self.transform_impl(**kwargs)
        self.post_transform(**kwargs)

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return self.input_type

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return self.output_type

    def set_input_uuid(self, input_uuid: str):
        """Set the Input UUID (Name) for this Transform"""
        self.input_uuid = input_uuid

    def set_output_uuid(self, output_uuid: str):
        """Set the Output UUID (Name) for this Transform"""
        self.output_uuid = output_uuid
