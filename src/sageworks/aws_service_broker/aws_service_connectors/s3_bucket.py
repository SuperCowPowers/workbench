"""S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""

import awswrangler as wr
from botocore.exceptions import ClientError


# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector
from sageworks.utils.aws_utils import compute_size


# Class to retrieve object/file information from an AWS S3 Bucket
class S3Bucket(Connector):
    """S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""

    def __init__(self, bucket: str):
        # Call SuperClass Initialization
        super().__init__()

        # Store our bucket name
        self.bucket = bucket
        self.s3_bucket_data = None

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            wr.s3.does_object_exist(self.bucket, boto3_session=self.boto3_session)
            return True
        except Exception:
            self.log.critical(f"Could not connect to AWS S3 {self.bucket}!")
            return False

    def refresh(self):
        """Refresh all the file/object data from this bucket"""
        self.log.debug(f"Refreshing S3 Bucket: {self.bucket}...")
        try:
            _aws_file_info = wr.s3.describe_objects(self.bucket, boto3_session=self.boto3_session)
        except ClientError as error:
            # If the exception is a ResourceNotFound, this is fine, otherwise raise all other exceptions
            if error.response["Error"]["Code"] in ["ResourceNotFound", "NoSuchBucket"]:
                self.log.warning(f"Describing objects in {self.bucket} gave ResourceNotFound")
                return {}
            else:
                self.log.warning(f"Describing objects in {self.bucket} gave {error.response['Error']['Code']}")
                return {}
        self.s3_bucket_data = {full_path: info for full_path, info in _aws_file_info.items()}

        # Track the size of the metadata
        for key in self.s3_bucket_data.keys():
            self.metadata_size_info[key] = compute_size(self.s3_bucket_data[key])

    def summary(self) -> dict:
        """Return a summary of all the file/objects in our bucket"""
        return self.s3_bucket_data

    def file_names(self) -> list:
        """Get all the file names in this bucket"""
        return list(self.s3_bucket_data.keys())

    def bucket_size(self) -> list:
        """For all the files in this bucket/prefix recursively SUM up the sizes"""
        sizes = []
        for file, info in self.s3_bucket_data.items():
            sizes.append(info["ContentLength"])
        return sum(sizes)


if __name__ == "__main__":
    """Exercises the S3Bucket Class"""
    from sageworks.utils.config_manager import ConfigManager
    from pprint import pprint

    # Grab out incoming data bucket for something to test with
    cm = ConfigManager()
    sageworks_bucket = cm.get_config("SAGEWORKS_BUCKET")
    incoming_data_bucket = "s3://" + sageworks_bucket + "/incoming-data/"

    # Create the class and check the functionality
    s3_bucket = S3Bucket(incoming_data_bucket)
    s3_bucket.check()
    s3_bucket.refresh()

    # Get the Summary Information
    pprint(s3_bucket.summary())

    # List the S3 Files
    print("S3 Objects:")
    for file_name in s3_bucket.file_names():
        print(f"\n*** {file_name} ***")

    # Get the size of all the objects in this bucket
    print(f"Bucket Size: {s3_bucket.bucket_size()}")

    # Print out the metadata sizes for this connector
    pprint(s3_bucket.get_metadata_sizes())

    # Test the functionality for a bucket that doesn't exist
    not_exist_bucket = "s3://non_existent_bucket"
    s3_bucket = S3Bucket(not_exist_bucket)
    s3_bucket.check()
    s3_bucket.refresh()
