"""S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""
import sys
import argparse
import awswrangler as wr


# Class to retrieve object/file information from an AWS S3 Bucket
class S3Bucket:
    """S3Bucket: Class to retrieve object/file information from an AWS S3 Bucket"""
    def __init__(self, bucket: str):
        # Store our bucket name
        self.bucket = bucket
        self.file_info = None

        # Load in the files from the bucket
        self.reload_files()

    def reload_files(self):
        """Load/reload the files in the bucket"""
        # Grab all the files in this bucket
        print(f"Reading S3 Bucket: {self.bucket}...")
        _aws_file_info = wr.s3.describe_objects(self.bucket)
        self.file_info = {full_path.split('/')[-1]: info for full_path, info in _aws_file_info.items()}

    def get_file_names(self) -> list:
        """Get all the file names in this bucket"""
        return list(self.file_info.keys())

    def get_file_info(self, file: str) -> dict:
        """Get additional info about this specific file"""
        return self.file_info[file]


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default='s3://sageworks-incoming-data', help='AWS S3 Bucket')
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Data Catalog database info
    s3_bucket = S3Bucket(args.bucket)

    # List files in the bucket
    print(f"{s3_bucket.bucket}")
    for file_name in s3_bucket.get_file_names():
        print(f"\t{file_name}")

    # Get additional info for a specific file
    my_file_info = s3_bucket.get_file_info('aqsol_public_data.csv')
    pprint(my_file_info)
