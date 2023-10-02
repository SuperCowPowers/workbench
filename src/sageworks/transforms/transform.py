"""Transform: Base Class for all transforms within SageWorks
              Inherited Classes must implement the abstract transform_impl() method"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import final
import os
import sys
import logging
import awswrangler as wr

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class TransformInput(Enum):
    """Enumerated Types for SageWorks Transform Inputs"""

    LOCAL_FILE = auto()
    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()


class TransformOutput(Enum):
    """Enumerated Types for SageWorks Transform Outputs"""

    PANDAS_DF = auto()
    SPARK_DF = auto()
    S3_OBJECT = auto()
    DATA_SOURCE = auto()
    FEATURE_SET = auto()
    MODEL = auto()
    ENDPOINT = auto()


class Transform(ABC):
    """Transform: Base Class for all transforms within SageWorks. Inherited Classes
    must implement the abstract transform_impl() method"""

    def __init__(self, input_uuid: str, output_uuid: str):
        """Transform Initialization"""

        self.log = logging.getLogger("sageworks")
        self.input_type = None
        self.output_type = None
        self.output_tags = ""
        self.input_uuid = str(input_uuid)  # Occasionally we get a pathlib.Path object
        self.output_uuid = str(output_uuid)  # Occasionally we get a pathlib.Path object
        self.output_meta = {"sageworks_input": self.input_uuid}
        self.data_catalog_db = "sageworks"

        # Grab our SageWorks Bucket
        self.sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
        if self.sageworks_bucket is None:
            print("Could not find ENV var for SAGEWORKS_BUCKET!")
            sys.exit(1)
        self.data_source_s3_path = "s3://" + self.sageworks_bucket + "/data-sources"
        self.feature_sets_s3_path = "s3://" + self.sageworks_bucket + "/feature-sets"

        # Grab a SageWorks Role ARN, Boto3, SageMaker Session, and SageMaker Client
        self.aws_account_clamp = AWSAccountClamp()
        self.sageworks_role_arn = self.aws_account_clamp.sageworks_execution_role_arn()
        self.boto_session = self.aws_account_clamp.boto_session()
        self.sm_session = self.aws_account_clamp.sagemaker_session(self.boto_session)
        self.sm_client = self.aws_account_clamp.sagemaker_client(self.boto_session)

        # Make sure the AWS data catalog database exists
        self.ensure_aws_catalog_db(self.data_catalog_db)
        self.ensure_aws_catalog_db("sagemaker_featurestore")

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

    def set_output_tags(self, tags: list | str):
        """Set the tags that will be associated with the output object
        Args:
            tags (list | str): The list of tags or a ':' separated string of tags"""
        if isinstance(tags, list):
            self.output_tags = ":".join(tags)
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
        sageworks_meta = {"sageworks_tags": self.output_tags}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value
        aws_tags = self.convert_to_aws_tags(sageworks_meta)
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

    def ensure_aws_catalog_db(self, catalog_db: str):
        """Ensure that the AWS Catalog Database exists"""
        wr.catalog.create_database(catalog_db, exist_ok=True, boto3_session=self.boto_session)
