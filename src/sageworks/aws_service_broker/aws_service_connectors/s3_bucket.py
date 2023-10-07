"""S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""
import os
import awswrangler as wr
from botocore.exceptions import ClientError


# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector


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
            wr.s3.does_object_exist(self.bucket, boto3_session=self.boto_session)
            return True
        except Exception as e:
            self.log.critical(f"Could not connect to AWS S3 {self.bucket}: {e}")
            return False

    def refresh_impl(self):
        """Load/reload the files in the bucket"""
        # Grab all the files in this bucket
        self.log.debug(f"Reading S3 Bucket: {self.bucket}...")
        try:
            _aws_file_info = wr.s3.describe_objects(self.bucket, boto3_session=self.boto_session)
        except ClientError as error:
            # If the exception is a ResourceNotFound, this is fine, otherwise raise all other exceptions
            if error.response["Error"]["Code"] in ["ResourceNotFound", "NoSuchBucket"]:
                self.log.warning(f"Describing objects in {self.bucket} gave ResourceNotFound")
                return {}
            else:
                self.log.warning(f"Describing objects in {self.bucket} gave {error.response['Error']['Code']}")
                return {}
        self.s3_bucket_data = {full_path: info for full_path, info in _aws_file_info.items()}

    def aws_meta(self) -> dict:
        """Return ALL the AWS metadata for the AWS S3 Service"""
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

    def file_info(self, file: str) -> dict:
        """Get additional info about this specific file"""
        return self.s3_bucket_data[file]


if __name__ == "__main__":
    """Exercises the S3Bucket Class"""
    import sys
    from pprint import pprint

    # Grab out incoming data bucket for something to test with
    sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
    if sageworks_bucket is None:
        print("Could not find ENV var for SAGEWORKS_BUCKET!")
        sys.exit(1)
    incoming_data_bucket = "s3://" + sageworks_bucket + "/incoming-data/"

    # Create the class and check the functionality
    s3_bucket = S3Bucket(incoming_data_bucket)
    s3_bucket.check()
    s3_bucket.refresh()

    # List files in the bucket
    print(f"{s3_bucket.bucket}")
    for file_name in s3_bucket.file_names():
        print(f"\t{file_name}")

    # Get the size of all the objects in this bucket
    print(f"Bucket Size: {s3_bucket.bucket_size()}")

    # Get additional info for a specific file
    my_file_info = s3_bucket.file_info(file_name)
    pprint(my_file_info)

    # Test the functionality for a bucket that doesn't exist
    not_exist_bucket = "s3://non_existent_bucket"
    s3_bucket = S3Bucket(not_exist_bucket)
    s3_bucket.check()
    s3_bucket.refresh()
