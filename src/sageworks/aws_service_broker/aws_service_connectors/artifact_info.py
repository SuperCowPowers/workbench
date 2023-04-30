"""ArtifactInfo: Class to retrieve information (tags, meta, size) from an AWS Artifact"""
import botocore
import awswrangler as wr


# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.cache import Cache


# Class to retrieve object/file information from an AWS S3 Bucket
class ArtifactInfo(Connector):
    """ArtifactInfo: Class to retrieve information (tags, meta, size) from an AWS Artifact"""

    def __init__(self):
        # Call SuperClass Initialization
        super().__init__()

        # Cache for information on AWS Artifacts (size, tags, metadata, etc)
        self.artifact_info_cache = Cache(expire=60)

    def check(self) -> bool:
        """Can we connect to this AWS Service?"""
        return True  # I'm great thx for asking!

    def refresh_impl(self):
        """Refresh the cache of AWS Artifact information"""
        # The cache will handle the refresh
        pass

    def metadata(self) -> dict:
        """Get all the metadata for this AWS connector"""
        return {}

    def s3_object_sizes(self, s3_path) -> int:
        """Return the sum of all the s3 objects sizes for the given s3 path
        Args:
            s3_path (str): S3 Path for recursive aggregation
        Returns:
            int: Sum of object size in this s3 path in MegaBytes
        """
        size_in_mb = self.artifact_info_cache.get(s3_path)
        if size_in_mb is None:
            self.log.info(f"Computing S3 Object sizes: {s3_path}...")
            size_in_bytes = sum(wr.s3.size_objects(s3_path, boto3_session=self.boto_session).values())
            size_in_mb = f"{ (size_in_bytes/1_000_000):.1f}"
            self.artifact_info_cache.set(s3_path, size_in_mb)
        return size_in_mb

    def get_sagemaker_obj_info(self, aws_arn) -> dict:
        """Retrieve information on AWS *SageMaker* Objects (tags, metadata, etc)
           This method will ONLY work for FeatureSets, Models, and Endpoints but fail for DataSources
        Args:
            aws_arn (str): AWS ARN for the artifact
        Returns:
            dict: Dictionary of AWS Artifact information
        """
        info = self.artifact_info_cache.get(aws_arn)
        if info is None:
            self.log.info(f"Retrieving Artifact Tags and Metadata: {aws_arn}...")

            # This class will work for FeatureSets, Models, and Endpoints but fail for DataSources
            try:
                aws_tags = self.sm_session.list_tags(aws_arn)
                info = self._aws_tags_to_dict(aws_tags)
            except botocore.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "ValidationException":
                    self.log.error(f"This method doesn't work on DataSources: {exc}")
                    return {}
                else:
                    self.log.critical(f"Unknown Error: {exc}")
                    raise exc

            # Set the artifact info cache
            self.artifact_info_cache.set(aws_arn, info)
        return info

    @staticmethod
    def _aws_tags_to_dict(aws_tags):
        """Internal: AWS Tags are in an odd format, so convert to regular dictionary"""
        return {item["Key"]: item["Value"] for item in aws_tags if "sageworks" in item["Key"]}


if __name__ == "__main__":
    from pprint import pprint
    from sageworks.artifacts.data_sources.athena_source import AthenaSource
    from sageworks.artifacts.feature_sets.feature_set import FeatureSet

    # Grab one of our test FeatureSets for something to test with
    feature_set = FeatureSet("abalone_features")
    details = feature_set.details()
    arn = details["aws_arn"]
    s3_location = details["s3_storage_location"]

    # Create the class and get the info about the artifact
    my_info = ArtifactInfo()
    meta = my_info.get_sagemaker_obj_info(arn)
    size = my_info.s3_object_sizes(s3_location)
    pprint(meta)
    print(f"Size: {size} MB")

    # Grab one of our test DataSources for something to test with
    data_source = AthenaSource("abalone_data")
    details = data_source.details()
    arn = details["aws_arn"]
    s3_location = details["s3_storage_location"]
