"""Artifact: Abstract Base Class for all Artifact classes in SageWorks.
                Artifacts simply reflect and aggregate one or more AWS Services"""
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import json

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.utils.sageworks_config import SageWorksConfig
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class Artifact(ABC):
    """Artifact: Abstract Base Class for all Artifact classes in SageWorks.
    Artifacts simply reflect and aggregate one or more AWS Services"""

    # Class attributes
    log = logging.getLogger(__name__)

    # Set up our Boto3 and SageMaker Session and SageMaker Client
    aws_account_clamp = AWSAccountClamp()
    boto_session = aws_account_clamp.boto_session()
    sm_session = aws_account_clamp.sagemaker_session(boto_session)
    sm_client = aws_account_clamp.sagemaker_client(boto_session)
    aws_region = aws_account_clamp.region()

    # AWSServiceBroker pulls and collects metadata from a bunch of AWS Services
    aws_broker = AWSServiceBroker()

    # Grab our SageWorksConfig for S3 Buckets and other SageWorks specific settings
    sageworks_config = SageWorksConfig()
    data_catalog_db = "sageworks"
    sageworks_bucket = sageworks_config.get_config_value("SAGEWORKS_AWS", "S3_BUCKET_NAME")
    data_source_s3_path = "s3://" + sageworks_bucket + "/data-sources"
    feature_sets_s3_path = "s3://" + sageworks_bucket + "/feature-sets"

    def __init__(self, uuid):
        """Artifact Initialization"""
        self.uuid = uuid
        self.log = logging.getLogger(__name__)

    @abstractmethod
    def check(self) -> bool:
        """Does the Artifact exist? Can we connect to it?"""
        pass

    @abstractmethod
    def details(self) -> dict:
        """Additional Details about this Artifact"""
        pass

    @abstractmethod
    def size(self) -> float:
        """Return the size of this artifact in MegaBytes"""
        pass

    @abstractmethod
    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        pass

    @abstractmethod
    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        pass

    @abstractmethod
    def arn(self):
        """AWS ARN (Amazon Resource Name) for this artifact"""
        pass

    @abstractmethod
    def aws_url(self):
        """AWS console/web interface for this artifact"""
        pass

    @abstractmethod
    def meta(self) -> dict:
        """Get the full AWS metadata for this artifact"""
        pass

    @abstractmethod
    def delete(self):
        """Delete this artifact including all related AWS objects"""
        pass

    def sageworks_meta(self) -> dict:
        """Get the SageWorks specific metadata for this Artifact
        Note: This functionality will work for FeatureSets, Models, and Endpoints
              but not for DataSources. DataSources need to overwrite this method
        """
        aws_arn = self.arn()
        self.log.info(f"Retrieving SageWorks Metadata for Artifact: {aws_arn}...")
        aws_tags = self.sm_session.list_tags(aws_arn)
        meta = self._aws_tags_to_dict(aws_tags)
        return meta

    def upsert_sageworks_meta(self, new_meta: dict):
        """Add SageWorks specific metadata to this Artifact
        Args:
            new_meta (dict): Dictionary of new metadata to add
        Note:
            This functionality will work for FeatureSets, Models, and Endpoints
            but not for DataSources. DataSources need to overwrite this method
        """
        aws_arn = self.arn()
        self.log.info(f"Upserting SageWorks Metadata for Artifact: {aws_arn}...")
        aws_tags = self._dict_to_aws_tags(new_meta)
        self.sm_session.sagemaker_client.add_tags(ResourceArn=aws_arn, Tags=aws_tags)

    def sageworks_tags(self) -> list:
        """Get the tags for this artifact"""
        combined_tags = self.sageworks_meta().get("sageworks_tags", "")
        tags = combined_tags.split(":")
        return tags

    def summary(self) -> dict:
        """This is generic summary information for all Artifacts. If you
        want to get more detailed information, call the details() method
        which is implemented by the specific Artifact class"""
        return {
            "uuid": self.uuid,
            "aws_arn": self.arn(),
            "aws_url": self.aws_url(),
            "size": self.size(),
            "created": self.created(),
            "modified": self.modified(),
            "sageworks_tags": self.sageworks_tags(),
            "sageworks_meta": self.sageworks_meta(),
        }

    @staticmethod
    def _aws_tags_to_dict(aws_tags) -> dict:
        """Internal: AWS Tags are in an odd format, so convert to regular dictionary"""
        return {item["Key"]: item["Value"] for item in aws_tags if "sageworks" in item["Key"]}

    @staticmethod
    def _dict_to_aws_tags(meta_data: dict) -> list:
        """Internal: AWS Tags are in an odd format, so we need to dictionary
        Args:
            meta_data (dict): Dictionary of metadata to convert to AWS Tags
        """

        # First convert any non-string values to JSON strings
        for key, value in meta_data.items():
            if not isinstance(value, str):
                meta_data[key] = json.dumps(value)

        # Now convert to AWS Tags format
        aws_tags = []
        for key, value in meta_data.items():
            aws_tags.append({"Key": key, "Value": value})
        return aws_tags
