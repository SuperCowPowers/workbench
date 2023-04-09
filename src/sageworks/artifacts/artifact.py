"""Artifact: Abstract Base Class for all Artifact classes in SageWorks.
                Artifacts simply reflect and aggregate one or more AWS Services"""
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class Artifact(ABC):
    """Artifact: Abstract Base Class for all Artifact classes in SageWorks.
    Artifacts simply reflect and aggregate one or more AWS Services"""

    # Class attributes
    log = logging.getLogger(__name__)

    # Set up our Boto3 and SageMaker Session and SageMaker Client
    boto_session = AWSAccountClamp().boto_session()
    sm_session = AWSAccountClamp().sagemaker_session()
    sm_client = AWSAccountClamp().sagemaker_client()

    # AWSServiceBroker pulls and collects metadata from a bunch of AWS Services
    aws_meta = AWSServiceBroker()

    def __init__(self, uuid):
        """Artifact Initialization"""
        self.uuid = uuid
        self.log = logging.getLogger(__name__)

        # FIXME: We should have this come from AWS or Config
        self.data_catalog_db = "sageworks"
        self.data_source_s3_path = "s3://scp-sageworks-artifacts/data-sources"
        self.feature_sets_s3_path = "s3://scp-sageworks-artifacts/feature-sets"

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
    def meta(self):
        """Get the full AWS metadata for this artifact"""
        pass

    @abstractmethod
    def delete(self):
        """Delete this artifact including all related AWS objects"""
        pass

    @staticmethod
    def aws_tags_to_dict(aws_tags):
        """AWS Tags are in an odd format, so convert to regular dictionary"""
        return {item["Key"]: item["Value"] for item in aws_tags if "sageworks" in item["Key"]}

    def sageworks_meta(self):
        """Get the SageWorks specific metadata for this Artifact"""
        aws_arn = self.arn()
        self.log.info(f"Retrieving SageWorks Metadata for Artifact: {aws_arn}...")
        aws_tags = self.sm_session.list_tags(aws_arn)
        meta = self.aws_tags_to_dict(aws_tags)
        return meta

    def sageworks_tags(self):
        """Get the tags for this artifact"""
        combined_tags = self.sageworks_meta().get("sageworks_tags", "")
        tags = combined_tags.split(":")
        return tags

    def add_tag(self, tag):
        """Add a tag to this artifact"""
        self.log.error("add_tag: functionality needs to be added!")

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
