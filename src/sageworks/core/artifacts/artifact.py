"""Artifact: Abstract Base Class for all Artifact classes in SageWorks.
                Artifacts simply reflect and aggregate one or more AWS Services"""
from abc import ABC, abstractmethod
from datetime import datetime
import os
import sys
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.utils.sageworks_cache import SageWorksCache
from sageworks.utils.aws_utils import list_tags_with_throttle, dict_to_aws_tags, sagemaker_delete_tag


class Artifact(ABC):
    """Artifact: Abstract Base Class for all Artifact classes in SageWorks.
    Artifacts simply reflect and aggregate one or more AWS Services"""

    # Class attributes

    # Set up our Boto3 and SageMaker Session and SageMaker Client
    aws_account_clamp = AWSAccountClamp()
    boto_session = aws_account_clamp.boto_session()
    sm_session = aws_account_clamp.sagemaker_session(boto_session)
    sm_client = aws_account_clamp.sagemaker_client(boto_session)
    aws_region = aws_account_clamp.region

    # AWSServiceBroker pulls and collects metadata from a bunch of AWS Services
    aws_broker = AWSServiceBroker()

    # Grab our SageWorks Bucket from ENV
    sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
    if sageworks_bucket is None:
        log = logging.getLogger("sageworks")
        log.critical("Could not find ENV var for SAGEWORKS_BUCKET!")
        sys.exit(1)

    # Setup Bucket Paths
    data_sources_s3_path = "s3://" + sageworks_bucket + "/data-sources"
    feature_sets_s3_path = "s3://" + sageworks_bucket + "/feature-sets"
    models_s3_path = "s3://" + sageworks_bucket + "/models"
    endpoints_s3_path = "s3://" + sageworks_bucket + "/endpoints"

    # Data Cache for Artifacts
    data_storage = SageWorksCache(prefix="data_storage")
    temp_storage = SageWorksCache(prefix="temp_storage", expire=300)  # 5 minutes
    ephemeral_storage = SageWorksCache(prefix="ephemeral_storage", expire=1)  # 1 second

    def __init__(self, uuid: str):
        """Artifact Initialization"""
        self.uuid = uuid
        self.log = logging.getLogger("sageworks")

    def __post_init__(self):
        """Artifact Post Initialization"""

        # Do I exist? (very metaphysical)
        if not self.exists():
            self.log.debug(f"Artifact {self.uuid} does not exist")
            return

        # Conduct a Health Check on this Artifact
        health_issues = self.health_check()
        if health_issues:
            if "needs_onboard" in health_issues:
                self.log.important(f"Artifact {self.uuid} needs to be onboarded")
            elif health_issues == ["no_activity"]:
                self.log.debug(f"Artifact {self.uuid} has no activity")
            else:
                self.log.warning(f"Health Check Failed {self.uuid}: {health_issues}")
            for issue in health_issues:
                self.add_sageworks_health_tag(issue)
        else:
            self.log.info(f"Health Check Passed {self.uuid}")

    @abstractmethod
    def exists(self) -> bool:
        """Does the Artifact exist? Can we connect to it?"""
        pass

    def sageworks_meta(self) -> dict:
        """Get the SageWorks specific metadata for this Artifact
        Note: This functionality will work for FeatureSets, Models, and Endpoints
              but not for DataSources. The DataSource (or child) class overrides this method.
        """
        # First, check our cache
        meta_data_key = f"{self.uuid}_sageworks_meta"
        meta_data = self.ephemeral_storage.get(meta_data_key)
        if meta_data is not None:
            return meta_data

        # Otherwise, fetch the metadata from AWS, store it in the cache, and return it
        meta_data = list_tags_with_throttle(self.arn(), self.sm_session)
        self.ephemeral_storage.set(meta_data_key, meta_data)
        return meta_data

    def expected_meta(self) -> list[str]:
        """Metadata we expect to see for this Artifact when it's ready
        Returns:
            list[str]: List of expected metadata keys
        """

        # If an artifact has additional expected metadata override this method
        return ["sageworks_status"]

    @abstractmethod
    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        pass

    def ready(self) -> bool:
        """Is the Artifact ready? Is initial setup complete and expected metadata populated?"""

        # If anything goes wrong, assume the artifact is not ready
        try:
            # Check for the expected metadata
            expected_meta = self.expected_meta()
            existing_meta = self.sageworks_meta()
            ready = set(existing_meta.keys()).issuperset(expected_meta)
            if ready:
                return True
            else:
                self.log.info("Artifact is not ready!")
                return False
        except Exception as e:
            self.log.error(f"Artifact malformed: {e}")
            return False

    @abstractmethod
    def make_ready(self) -> bool:
        """Make the Artifact ready. Perform any initial setup steps"""
        pass

    def onboard(self) -> bool:
        """Onboard this Model into SageWorks
        Returns:
            bool: True if the Model was successfully onboarded, False otherwise
        """
        self.log.important(f"Onboarding {self.uuid}...")
        self.set_status("onboarding")
        self.make_ready()
        return True

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
    def aws_meta(self) -> dict:
        """Get the full AWS metadata for this artifact"""
        pass

    @abstractmethod
    def delete(self):
        """Delete this artifact including all related AWS objects"""
        pass

    def upsert_sageworks_meta(self, new_meta: dict):
        """Add SageWorks specific metadata to this Artifact
        Args:
            new_meta (dict): Dictionary of NEW metadata to add
        Note:
            This functionality will work for FeatureSets, Models, and Endpoints
            but not for DataSources. The DataSource class overrides this method.
        """
        # Sanity check
        aws_arn = self.arn()
        if aws_arn is None:
            self.log.error(f"ARN is None for {self.uuid}!")
            return

        # Add the new metadata to the existing metadata
        self.log.info(f"Upserting SageWorks Metadata for Artifact: {aws_arn}...")
        aws_tags = dict_to_aws_tags(new_meta)
        self.sm_client.add_tags(ResourceArn=aws_arn, Tags=aws_tags)

    def remove_sageworks_meta(self, key_to_remove: str):
        """Remove SageWorks specific metadata from this Artifact
        Args:
            key_to_remove (str): The metadata key to remove
        Note:
            This functionality will work for FeatureSets, Models, and Endpoints
            but not for DataSources. The DataSource class overrides this method.
        """
        aws_arn = self.arn()
        # Sanity check
        if aws_arn is None:
            self.log.error(f"ARN is None for {self.uuid}!")
            return
        self.log.info(f"Removing SageWorks Metadata {key_to_remove} for Artifact: {aws_arn}...")
        sagemaker_delete_tag(aws_arn, self.sm_session, key_to_remove)

    def sageworks_tags(self, tag_type="user") -> list:
        """Get the tags for this artifact
        Args:
            tag_type (str): Type of tags to return (user or health)
        Returns:
            list[str]: List of tags for this artifact
        """
        if tag_type == "user":
            user_tags = self.sageworks_meta().get("sageworks_tags")
            return user_tags.split(":") if user_tags else []

        # Grab our health tags
        health_tags = self.sageworks_meta().get("sageworks_health_tags")

        # If we don't have health tags, create the storage and return an empty list
        if health_tags is None:
            self.log.important(f"{self.uuid} creating sageworks_health_tags storage...")
            self.upsert_sageworks_meta({"sageworks_health_tags": ""})
            return []

        # Otherwise, return the health tags
        return health_tags.split(":") if health_tags else []

    def add_sageworks_tag(self, tag, tag_type="user"):
        """Add a tag for this artifact, ensuring no duplicates and maintaining order.
        Args:
            tag (str): Tag to add for this artifact
            tag_type (str): Type of tag to add (user or health)
        """
        current_tags = self.sageworks_tags(tag_type) if tag_type == "user" else self.sageworks_health_tags()
        if tag not in current_tags:
            current_tags.append(tag)
            combined_tags = ":".join(current_tags)
            if tag_type == "user":
                self.upsert_sageworks_meta({"sageworks_tags": combined_tags})
            else:
                self.upsert_sageworks_meta({"sageworks_health_tags": combined_tags})

    def remove_sageworks_tag(self, tag, tag_type="user"):
        """Remove a tag from this artifact if it exists.
        Args:
            tag (str): Tag to remove from this artifact
            tag_type (str): Type of tag to remove (user or health)
        """
        current_tags = self.sageworks_tags(tag_type) if tag_type == "user" else self.sageworks_health_tags()
        if tag in current_tags:
            current_tags.remove(tag)
            combined_tags = ":".join(current_tags)
            if tag_type == "user":
                self.upsert_sageworks_meta({"sageworks_tags": combined_tags})
            elif tag_type == "health":
                self.upsert_sageworks_meta({"sageworks_health_tags": combined_tags})

    # Syntactic sugar for health tags
    def sageworks_health_tags(self):
        return self.sageworks_tags(tag_type="health")

    def set_sageworks_health_tags(self, tags):
        self.upsert_sageworks_meta({"sageworks_health_tags": ":".join(tags)})

    def add_sageworks_health_tag(self, tag):
        self.add_sageworks_tag(tag, tag_type="health")

    def remove_sageworks_health_tag(self, tag):
        self.remove_sageworks_tag(tag, tag_type="health")

    def get_input(self) -> str:
        """Get the input data for this artifact"""
        return self.sageworks_meta().get("sageworks_input", "unknown")

    def get_status(self) -> str:
        """Get the status for this artifact"""
        return self.sageworks_meta().get("sageworks_status", "unknown")

    def set_status(self, status: str):
        """Set the status for this artifact
        Args:
            status (str): Status to set for this artifact
        """
        self.upsert_sageworks_meta({"sageworks_status": status})

    def health_check(self) -> list[str]:
        """Perform a health check on this artifact
        Returns:
            list[str]: List of health issues
        """
        health_issues = []
        if not self.ready():
            return ["needs_onboard"]
        if "unknown" in self.aws_url():
            health_issues.append("aws_url_unknown")
        return health_issues

    def summary(self) -> dict:
        """This is generic summary information for all Artifacts. If you
        want to get more detailed information, call the details() method
        which is implemented by the specific Artifact class"""
        return {
            "uuid": self.uuid,
            "health_tags": self.sageworks_health_tags(),
            "aws_arn": self.arn(),
            "size": self.size(),
            "created": self.created(),
            "modified": self.modified(),
            "input": self.get_input(),
            "sageworks_tags": self.sageworks_tags(),
        }

    def delete_metadata(self, key_to_delete: str):
        """Delete specific metadata from this artifact
        Args:
            key_to_delete (str): Metadata key to delete
        """

        aws_arn = self.arn()
        self.log.important(f"Deleting Metadata {key_to_delete} for Artifact: {aws_arn}...")

        # First, fetch all the existing tags
        response = self.sm_session.list_tags(aws_arn)
        existing_tags = response.get("Tags", [])

        # Convert existing AWS tags to a dictionary for easy manipulation
        existing_tags_dict = {item["Key"]: item["Value"] for item in existing_tags}

        # Identify tags to delete
        tag_list_to_delete = []
        for key in existing_tags_dict.keys():
            if key == key_to_delete or key.startswith(f"{key_to_delete}_chunk_"):
                tag_list_to_delete.append(key)

        # Delete the identified tags
        if tag_list_to_delete:
            self.sm_client.delete_tags(ResourceArn=aws_arn, TagKeys=tag_list_to_delete)
        else:
            self.log.info(f"No Metadata found: {key_to_delete}...")


if __name__ == "__main__":
    """Exercise the Artifact Class"""
    from sageworks.api.data_source import DataSource
    from sageworks.api.feature_set import FeatureSet

    # Create a DataSource (which is a subclass of Artifact)
    data_source = DataSource("test_data")

    # Just some random tests
    assert data_source.exists()

    print(f"UUID: {data_source.uuid}")
    print(f"Ready: {data_source.ready()}")
    print(f"Status: {data_source.get_status()}")
    print(f"Input: {data_source.get_input()}")

    # Create a FeatureSet (which is a subclass of Artifact)
    fs = FeatureSet("test_features")

    # Just some random tests
    assert fs.exists()

    print(f"UUID: {fs.uuid}")
    print(f"Ready: {fs.ready()}")
    print(f"Status: {fs.get_status()}")
    print(f"Input: {fs.get_input()}")
