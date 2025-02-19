"""Artifact: Abstract Base Class for all Artifact classes in Workbench.
Artifacts simply reflect and aggregate one or more AWS Services"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Union

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.core.cloud_platform.aws.aws_df_store import AWSDFStore as DFStore
from workbench.utils.aws_utils import dict_to_aws_tags
from workbench.utils.config_manager import ConfigManager, FatalConfigError
from workbench.core.cloud_platform.cloud_meta import CloudMeta
from workbench.cached.cached_meta import CachedMeta


class Artifact(ABC):
    """Artifact: Abstract Base Class for all Artifact classes in Workbench"""

    # Class-level shared resources
    log = logging.getLogger("workbench")

    # Config Manager
    cm = ConfigManager()
    if not cm.config_okay():
        log = logging.getLogger("workbench")
        log.critical("Workbench Configuration Incomplete...")
        log.critical("Run the 'workbench' command and follow the prompts...")
        raise FatalConfigError()

    # AWS Account Clamp
    aws_account_clamp = AWSAccountClamp()
    boto3_session = aws_account_clamp.boto3_session
    sm_session = aws_account_clamp.sagemaker_session()
    sm_client = aws_account_clamp.sagemaker_client()
    aws_region = aws_account_clamp.region

    # Setup Bucket Paths
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
    data_sources_s3_path = f"s3://{workbench_bucket}/data-sources"
    feature_sets_s3_path = f"s3://{workbench_bucket}/feature-sets"
    models_s3_path = f"s3://{workbench_bucket}/models"
    endpoints_s3_path = f"s3://{workbench_bucket}/endpoints"

    # Delimiter for storing lists in AWS Tags
    tag_delimiter = "::"

    # Grab our Dataframe Storage
    df_cache = DFStore(path_prefix="/workbench/dataframe_cache")

    def __init__(self, uuid: str, use_cached_meta: bool = False):
        """Initialize the Artifact Base Class

        Args:
            uuid (str): The UUID of this artifact
            use_cached_meta (bool): Should we use cached metadata? (default: False)
        """
        self.uuid = uuid
        if use_cached_meta:
            self.log.info(f"Using Cached Metadata for {self.uuid}")
            self.meta = CachedMeta()
        else:
            self.meta = CloudMeta()

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
                self.log.debug(f"Artifact {self.uuid} has no activity, which is fine")
            else:
                self.log.warning(f"Health Check Failed {self.uuid}: {health_issues}")
            for issue in health_issues:
                self.add_health_tag(issue)
        else:
            self.log.info(f"Health Check Passed {self.uuid}")

    @classmethod
    def is_name_valid(cls, name: str, delimiter: str = "_", lower_case: bool = True) -> bool:
        """Check if the name adheres to the naming conventions for this Artifact.

        Args:
            name (str): The name/id to check.
            delimiter (str): The delimiter to use in the name/id string (default: "_")
            lower_case (bool): Should the name be lowercased? (default: True)

        Returns:
            bool: True if the name is valid, False otherwise.
        """
        valid_name = cls.generate_valid_name(name, delimiter=delimiter, lower_case=lower_case)
        if name != valid_name:
            cls.log.warning(f"Artifact name: '{name}' is not valid. Convert it to something like: '{valid_name}'")
            return False
        return True

    @staticmethod
    def generate_valid_name(name: str, delimiter: str = "_", lower_case: bool = True) -> str:
        """Only allow letters and the specified delimiter, also lowercase the string.

        Args:
            name (str): The name/id string to check.
            delimiter (str): The delimiter to use in the name/id string (default: "_")
            lower_case (bool): Should the name be lowercased? (default: True)

        Returns:
            str: A generated valid name/id.
        """
        valid_name = "".join(c for c in name if c.isalnum() or c in ["_", "-"])
        if lower_case:
            valid_name = valid_name.lower()

        # Replace with the chosen delimiter
        return valid_name.replace("_", delimiter).replace("-", delimiter)

    @abstractmethod
    def exists(self) -> bool:
        """Does the Artifact exist? Can we connect to it?"""
        pass

    def workbench_meta(self) -> Union[dict, None]:
        """Get the Workbench specific metadata for this Artifact

        Returns:
            Union[dict, None]: Dictionary of Workbench metadata for this Artifact

        Note: This functionality will work for FeatureSets, Models, and Endpoints
              but not for DataSources and Graphs, those classes need to override this method.
        """
        return self.meta.get_aws_tags(self.arn())

    def expected_meta(self) -> list[str]:
        """Metadata we expect to see for this Artifact when it's ready
        Returns:
            list[str]: List of expected metadata keys
        """

        # If an artifact has additional expected metadata override this method
        return ["workbench_status"]

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
            existing_meta = self.workbench_meta()
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
    def onboard(self) -> bool:
        """Onboard this Artifact into Workbench
        Returns:
            bool: True if the Artifact was successfully onboarded, False otherwise
        """
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
    def hash(self) -> str:
        """Return the hash of this artifact, useful for content validation"""
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

    def upsert_workbench_meta(self, new_meta: dict):
        """Add Workbench specific metadata to this Artifact
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
        self.log.info(f"Adding Tags to {self.uuid}:{str(new_meta)[:50]}...")
        aws_tags = dict_to_aws_tags(new_meta)
        try:
            self.sm_client.add_tags(ResourceArn=aws_arn, Tags=aws_tags)
        except Exception as e:
            self.log.error(f"Error adding metadata to {aws_arn}: {e}")

    def get_tags(self, tag_type="user") -> list:
        """Get the tags for this artifact
        Args:
            tag_type (str): Type of tags to return (user or health)
        Returns:
            list[str]: List of tags for this artifact
        """
        if tag_type == "user":
            user_tags = self.workbench_meta().get("workbench_tags")
            return user_tags.split(self.tag_delimiter) if user_tags else []

        # Grab our health tags
        health_tags = self.workbench_meta().get("workbench_health_tags")

        # If we don't have health tags, create the storage and return an empty list
        if health_tags is None:
            self.log.important(f"{self.uuid} creating workbench_health_tags storage...")
            self.upsert_workbench_meta({"workbench_health_tags": ""})
            return []

        # Otherwise, return the health tags
        return health_tags.split(self.tag_delimiter) if health_tags else []

    def set_tags(self, tags):
        self.upsert_workbench_meta({"workbench_tags": self.tag_delimiter.join(tags)})

    def add_tag(self, tag, tag_type="user"):
        """Add a tag for this artifact, ensuring no duplicates and maintaining order.
        Args:
            tag (str): Tag to add for this artifact
            tag_type (str): Type of tag to add (user or health)
        """
        current_tags = self.get_tags(tag_type) if tag_type == "user" else self.get_health_tags()
        if tag not in current_tags:
            current_tags.append(tag)
            combined_tags = self.tag_delimiter.join(current_tags)
            if tag_type == "user":
                self.upsert_workbench_meta({"workbench_tags": combined_tags})
            else:
                self.upsert_workbench_meta({"workbench_health_tags": combined_tags})

    def remove_workbench_tag(self, tag, tag_type="user"):
        """Remove a tag from this artifact if it exists.
        Args:
            tag (str): Tag to remove from this artifact
            tag_type (str): Type of tag to remove (user or health)
        """
        current_tags = self.get_tags(tag_type) if tag_type == "user" else self.get_health_tags()
        if tag in current_tags:
            current_tags.remove(tag)
            combined_tags = self.tag_delimiter.join(current_tags)
            if tag_type == "user":
                self.upsert_workbench_meta({"workbench_tags": combined_tags})
            elif tag_type == "health":
                self.upsert_workbench_meta({"workbench_health_tags": combined_tags})

    # Syntactic sugar for health tags
    def get_health_tags(self):
        return self.get_tags(tag_type="health")

    def set_health_tags(self, tags):
        self.upsert_workbench_meta({"workbench_health_tags": self.tag_delimiter.join(tags)})

    def add_health_tag(self, tag):
        self.add_tag(tag, tag_type="health")

    def remove_health_tag(self, tag):
        self.remove_workbench_tag(tag, tag_type="health")

    # Owner of this artifact
    def get_owner(self) -> str:
        """Get the owner of this artifact"""
        return self.workbench_meta().get("workbench_owner", "unknown")

    def set_owner(self, owner: str):
        """Set the owner of this artifact

        Args:
            owner (str): Owner to set for this artifact
        """
        self.upsert_workbench_meta({"workbench_owner": owner})

    def get_input(self) -> str:
        """Get the input data for this artifact"""
        return self.workbench_meta().get("workbench_input", "unknown")

    def set_input(self, input_data: str):
        """Set the input data for this artifact

        Args:
            input_data (str): Name of input data for this artifact
        Note:
            This breaks the official provenance of the artifact, so use with caution.
        """
        self.log.important(f"{self.uuid}: Setting input to {input_data}...")
        self.log.important("Be careful with this! It breaks automatic provenance of the artifact!")
        self.upsert_workbench_meta({"workbench_input": input_data})

    def get_status(self) -> str:
        """Get the status for this artifact"""
        return self.workbench_meta().get("workbench_status", "unknown")

    def set_status(self, status: str):
        """Set the status for this artifact
        Args:
            status (str): Status to set for this artifact
        """
        self.upsert_workbench_meta({"workbench_status": status})

    def health_check(self) -> list[str]:
        """Perform a health check on this artifact
        Returns:
            list[str]: List of health issues
        """
        health_issues = []
        if not self.ready():
            return ["needs_onboard"]
        # FIXME: Revisit AWS URL check
        # if "unknown" in self.aws_url():
        #    health_issues.append("aws_url_unknown")
        return health_issues

    def summary(self) -> dict:
        """This is generic summary information for all Artifacts. If you
        want to get more detailed information, call the details() method
        which is implemented by the specific Artifact class"""
        basic = {
            "uuid": self.uuid,
            "health_tags": self.get_health_tags(),
            "aws_arn": self.arn(),
            "size": self.size(),
            "created": self.created(),
            "modified": self.modified(),
            "input": self.get_input(),
        }
        # Combine the workbench metadata with the basic metadata
        return {**basic, **self.workbench_meta()}

    def __repr__(self) -> str:
        """String representation of this artifact

        Returns:
            str: String representation of this artifact
        """

        # If the artifact does not exist, return a message
        if not self.exists():
            return f"{self.__class__.__name__}: {self.uuid} does not exist"

        summary_dict = self.summary()
        display_keys = [
            "aws_arn",
            "health_tags",
            "size",
            "created",
            "modified",
            "input",
            "workbench_status",
            "workbench_tags",
        ]
        summary_items = [f"  {repr(key)}: {repr(value)}" for key, value in summary_dict.items() if key in display_keys]
        summary_str = f"{self.__class__.__name__}: {self.uuid}\n" + ",\n".join(summary_items)
        return summary_str

    def delete_metadata(self, key_to_delete: str):
        """Delete specific metadata from this artifact
        Args:
            key_to_delete (str): Metadata key to delete
        """

        aws_arn = self.arn()
        self.log.important(f"Deleting Metadata {key_to_delete} for Artifact: {aws_arn}...")

        # First, fetch all the existing tags
        existing_tags = self.sm_session.list_tags(aws_arn)

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
    from workbench.api.data_source import DataSource
    from workbench.api.feature_set import FeatureSet

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

    # Test new input method
    fs.set_input("test_data")
    print(f"Input: {fs.get_input()}")

    # Test add metadata
    fs.upsert_workbench_meta({"test_key": "test_value"})

    # Test delete metadata
    fs.delete_metadata("test_key")
