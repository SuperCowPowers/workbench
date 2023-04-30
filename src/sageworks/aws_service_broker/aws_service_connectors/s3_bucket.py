"""S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""
import awswrangler as wr


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
        self.log.info(f"Reading S3 Bucket: {self.bucket}...")
        _aws_file_info = wr.s3.describe_objects(self.bucket, boto3_session=self.boto_session)
        self.s3_bucket_data = {full_path.split("/")[-1]: info for full_path, info in _aws_file_info.items()}

    def metadata(self) -> dict:
        """Get all the metadata for the files in this bucket"""
        return self.s3_bucket_data

    def file_names(self) -> list:
        """Get all the file names in this bucket"""
        return list(self.s3_bucket_data.keys())

    def file_info(self, file: str) -> dict:
        """Get additional info about this specific file"""
        return self.s3_bucket_data[file]


if __name__ == "__main__":
    """Exercises the S3Bucket Class"""
    from pprint import pprint
    from sageworks.utils.sageworks_config import SageWorksConfig

    # Grab out incoming data bucket for something to test with
    sageworks_config = SageWorksConfig()
    sageworks_bucket = sageworks_config.get_config_value("SAGEWORKS_AWS", "S3_BUCKET_NAME")
    incoming_data_bucket = "s3://" + sageworks_bucket + "/incoming-data"

    # Create the class and check the functionality
    s3_bucket = S3Bucket(incoming_data_bucket)
    s3_bucket.check()
    s3_bucket.refresh()

    # List files in the bucket
    print(f"{s3_bucket.bucket}")
    for file_name in s3_bucket.file_names():
        print(f"\t{file_name}")

    # Get additional info for a specific file
    my_file_info = s3_bucket.file_info("abalone.csv")
    pprint(my_file_info)
