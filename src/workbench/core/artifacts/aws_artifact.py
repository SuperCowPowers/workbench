"""AWSArtifact: Base Class for all AWS-backed Artifact classes in Workbench.

Backs the Artifact metadata contract with AWS tags and provides the shared
AWS session/bucket resources used by every AWS artifact class.
"""

from abc import abstractmethod
import logging
from typing import Union
from botocore.exceptions import ClientError

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.core.df_store_core import DFStoreCore
from workbench.core.parameter_store_core import ParameterStoreCore as ParameterStore
from workbench.utils.aws_utils import aws_throttle, dict_to_aws_tags
from sagemaker.core.resources import Tag
from workbench.utils.config_manager import ConfigManager, FatalConfigError
from workbench.core.cloud_platform.cloud_meta import CloudMeta


class AWSArtifact(Artifact):
    """AWSArtifact: Base Class for all AWS-backed Artifact classes in Workbench"""

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
    # Scratch root for transient files, separate from the protected artifact
    # prefixes. Each use owns a subfolder (temp/training_data/, temp/athena_output/).
    temp_s3_path = f"s3://{workbench_bucket}/temp"

    # Grab our Dataframe Cache Storage (use the endpoint-safe core class directly
    # with our refreshable session + config-loaded bucket — equivalent to going
    # through workbench.api.DFStore but without triggering workbench.api.__init__
    # while artifact.py is still loading).
    df_cache = DFStoreCore(
        path_prefix="/workbench/dataframe_cache",
        s3_bucket=workbench_bucket,
        boto3_session=boto3_session,
    )

    # Artifact may want to use the Parameter Store or Dataframe Store
    param_store = ParameterStore(boto3_session=boto3_session)
    df_store = DFStoreCore(s3_bucket=workbench_bucket, boto3_session=boto3_session)

    def __init__(self, name: str, **kwargs):
        """Initialize the AWSArtifact Base Class

        Args:
            name (str): The Name of this artifact
        """
        super().__init__(name, **kwargs)
        self.meta = CloudMeta()

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

    def summary(self) -> dict:
        """Generic summary information, plus the artifact's ARN"""
        return {**super().summary(), "aws_arn": self.arn()}

    def workbench_meta(self) -> Union[dict, None]:
        """Get the Workbench specific metadata for this Artifact

        Returns:
            Union[dict, None]: Dictionary of Workbench metadata for this Artifact

        Note: This functionality will work for FeatureSets, Models, and Endpoints
              but not for DataSources and Graphs, those classes need to override this method.
        """
        return self.meta.get_aws_tags(self.arn())

    @aws_throttle
    def upsert_workbench_meta(self, new_meta: dict):
        """Add Workbench specific metadata to this Artifact
        Args:
            new_meta (dict): Dictionary of NEW metadata to add
        Note:
            This functionality will work for FeatureSets, Models, and Endpoints
            but not for DataSources. The DataSource class overrides this method.
        """

        # Check for ReadOnly Role
        if self.aws_account_clamp.read_only:
            self.log.info("Cannot add metadata with a ReadOnly Permissions...")
            return

        # Sanity check
        aws_arn = self.arn()
        if aws_arn is None:
            self.log.error(f"ARN is None for {self.name}!")
            return

        # Add the new metadata to the existing metadata
        self.log.info(f"Adding Tags to {self.name}:{str(new_meta)[:50]}...")
        aws_tags = dict_to_aws_tags(new_meta)
        try:
            Tag.add_tags(resource_arn=aws_arn, tags=aws_tags, session=self.boto3_session)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                raise  # @aws_throttle handles the backoff/retry
            self.log.error(f"Error adding metadata to {aws_arn}: {type(e).__name__}: {e}")
            return
        except Exception as e:
            self.log.error(f"Error adding metadata to {aws_arn}: {type(e).__name__}: {e}")
            return

        # Poke the modified registry so caches know this artifact changed
        from workbench.cached.cached_meta import CachedMeta

        CachedMeta().update_modified_timestamp(self)

    @aws_throttle
    def delete_metadata(self, key_to_delete: str):
        """Delete specific metadata from this artifact
        Args:
            key_to_delete (str): Metadata key to delete
        """

        aws_arn = self.arn()
        self.log.important(f"Deleting Metadata {key_to_delete} for Artifact: {aws_arn}...")

        # First, fetch all the existing tags using V3 API
        from sagemaker.core.common_utils import list_tags as sm_list_tags

        existing_tags = sm_list_tags(self.sm_session, aws_arn)

        # Convert existing AWS tags to a dictionary for easy manipulation
        existing_tags_dict = {item["Key"]: item["Value"] for item in existing_tags}

        # Identify tags to delete
        tag_list_to_delete = []
        for key in existing_tags_dict.keys():
            if key == key_to_delete or key.startswith(f"{key_to_delete}_chunk_"):
                tag_list_to_delete.append(key)

        # Delete the identified tags using V3 API
        if tag_list_to_delete:
            Tag.delete_tags(resource_arn=aws_arn, tag_keys=tag_list_to_delete, session=self.boto3_session)
        else:
            self.log.info(f"No Metadata found: {key_to_delete}...")


if __name__ == "__main__":
    """Exercise the Artifact Class"""
    from workbench.api import DataSource, FeatureSet, Endpoint

    # Grab an Endpoint (which is a subclass of Artifact)
    Endpoint("wine-classification")

    # Grab a DataSource (which is a subclass of Artifact)
    data_source = DataSource("test_data")

    # Just some random tests
    assert data_source.exists()

    print(f"Name: {data_source.name}")
    print(f"Ready: {data_source.ready()}")
    print(f"Status: {data_source.get_status()}")
    print(f"Input: {data_source.get_input()}")

    # Create a FeatureSet (which is a subclass of Artifact)
    fs = FeatureSet("test_features")

    # Just some random tests
    assert fs.exists()

    print(f"Name: {fs.name}")
    print(f"Ready: {fs.ready()}")
    print(f"Status: {fs.get_status()}")
    print(f"Input: {fs.get_input()}")

    # Test add metadata
    fs.upsert_workbench_meta({"test_key": "test_value"})

    # Test delete metadata
    fs.delete_metadata("test_key")
