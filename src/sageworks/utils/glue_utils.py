"""AWS Glue Job Utilities"""

import sys
from typing import List
import awswrangler as wr


def glue_args_to_dict(argv: list[str]) -> dict:
    """Take the Glue Jobs argv list of args and organize them into a dictionary

    Args:
        argv (list[str]): The Glue Jobs argv list of args

    Returns:
        dict: The Glue Jobs argv list of args organized into a dictionary
    """
    it = iter(argv[1:])
    return dict(zip(it, it))


def list_s3_files(s3_path: str, extensions: str = "*.csv") -> List[str]:
    """
    Lists files in an S3 path with specified extension.

    Args:
    s3_path (str): The full S3 path (e.g., 's3://my-bucket/my-prefix/').
    extensions (str): File extension to filter by, defaults to '*.csv'.

    Returns:
    List[str]: A list of file paths matching the extension in the S3 path.
    """
    files = wr.s3.list_objects(path=s3_path, suffix=extensions.lstrip("*"))
    return files


if __name__ == "__main__":
    # Test the glue utils functions
    print("Testing Glue Utils Functions")
    print(glue_args_to_dict(sys.argv))
    print(list_s3_files("s3://sageworks-public-data/common"))
