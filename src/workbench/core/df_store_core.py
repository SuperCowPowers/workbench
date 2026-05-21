"""DFStoreCore: Endpoint-safe storage of DataFrames using AWS S3/Parquet/Snappy.

This is the lightweight core class. The orchestration-side wrapper is
:class:`workbench.core.artifacts.df_store_core.DFStoreCore` (note: same
class name, different module — the artifacts version subclasses this one
and supplies a ``ConfigManager`` + ``AWSAccountClamp`` session). Endpoint
code that needs DataFrame storage can instantiate this directly with
explicit ``s3_bucket`` / ``boto3_session`` args, or fall back to env vars.
"""

from datetime import datetime
from typing import Union
import logging
import uuid
import awswrangler as wr
import pandas as pd
import os
import re
from urllib.parse import urlparse

# Workbench Imports
from workbench.core.cloud_platform.aws.boto_session import get_boto3_session
from workbench.core.parameter_store_core import ParameterStoreCore
from workbench.utils.aws_utils import not_found_returns_none


class DFStoreCore:
    """DFStoreCore: Endpoint-safe storage of DataFrames using AWS S3/Parquet/Snappy.

    Common Usage:
        ```python
        df_store = DFStoreCore()

        # List Data
        df_store.list()

        # Add DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df_store.upsert("/test/my_data", df)

        # Retrieve DataFrame
        df = df_store.get("/test/my_data")
        print(df)

        # Delete Data
        df_store.delete("/test/my_data")
        ```
    """

    def __init__(self, path_prefix: Union[str, None] = None, s3_bucket: Union[str, None] = None, boto3_session=None):
        """DFStoreCore Init Method

        Args:
            path_prefix (Union[str, None], optional): Path prefix for storage locations (Defaults to None)
            s3_bucket (Union[str, None], optional): S3 Bucket to use (Defaults to None, uses Workbench bucket)
            boto3_session (optional): Boto3 session to use (Defaults to None, uses Workbench session)
        """
        self.log = logging.getLogger("workbench")

        # Set up path prefix
        self._base_prefix = "df_store/"
        self.path_prefix = re.sub(r"/+", "/", self._base_prefix + (path_prefix or ""))

        # Resolve bucket: explicit arg > env var > parameter store
        self.workbench_bucket = (
            s3_bucket
            or os.getenv("WORKBENCH_BUCKET")
            or ParameterStoreCore().get("/workbench/config/workbench_bucket")
        )
        if not self.workbench_bucket:
            raise ValueError(
                "S3 bucket not found. Set WORKBENCH_BUCKET ENV or '/workbench/config/workbench_bucket' Parameter Store."
            )

        # Set up boto3 session and S3 client
        self.boto3_session = boto3_session or get_boto3_session()
        self.s3_client = self.boto3_session.client("s3")

    def list(self, prefix: str = None, include_cache: bool = False) -> list:
        """List all objects in the data_store prefix

        Args:
            prefix (str, optional): Only include objects with the given prefix
            include_cache (bool, optional): Include cache objects in the list (Defaults to False)

        Returns:
            list: A list of all the objects in the data_store prefix.
        """
        df = self.summary(include_cache=include_cache)
        if prefix:
            df = df[df["location"].str.startswith(prefix)]
        return df["location"].tolist()

    def last_modified(self, location: str) -> Union[datetime, None]:
        """Return the last modified date of a graph.

        Args:
            location (str): Logical location of the graph.

        Returns:
            Union[datetime, None]: Last modified datetime or None if not found.
        """
        s3_uri = self._generate_s3_uri(location)
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response["LastModified"]
        except self.s3_client.exceptions.ClientError:
            return None

    def summary(self, include_cache: bool = False) -> pd.DataFrame:
        """Return a nicely formatted summary of object locations, sizes (in MB), and modified dates.

        Args:
            include_cache (bool, optional): Include cache objects in the summary (Defaults to False)
        """
        df = self.details(include_cache=include_cache)

        # Create a formatted DataFrame
        formatted_df = pd.DataFrame(
            {
                "location": df["location"],
                "size (MB)": (df["size"] / (1024 * 1024)).round(2),  # Convert size to MB
                "modified": pd.to_datetime(df["modified"]).dt.strftime("%Y-%m-%d %H:%M:%S"),  # Format date
            }
        )
        return formatted_df

    def details(self, include_cache: bool = False) -> pd.DataFrame:
        """Return detailed metadata for all objects, optionally excluding the specified prefix.

        Args:
            include_cache (bool, optional): Include cache objects in the details (Defaults to False)
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.workbench_bucket, Prefix=self.path_prefix)
            if "Contents" not in response:
                return pd.DataFrame(columns=["location", "s3_file", "size", "modified"])

            # Collect details for each object
            data = []
            for obj in response["Contents"]:
                full_key = obj["Key"]

                # Reverse logic: Strip the bucket/prefix in the front and .parquet in the end
                location = full_key.replace(f"{self.path_prefix}", "/").split(".parquet")[0]
                s3_file = f"s3://{self.workbench_bucket}/{full_key}"
                size = obj["Size"]
                modified = obj["LastModified"]
                data.append([location, s3_file, size, modified])

            # Create the DataFrame
            df = pd.DataFrame(data, columns=["location", "s3_file", "size", "modified"])

            # Apply the exclude_prefix filter if set
            cache_prefix = "/workbench/dataframe_cache/"
            if not include_cache:
                df = df[~df["location"].str.startswith(cache_prefix)]

            return df

        except Exception as e:
            self.log.error(f"Failed to get object details: {e}")
            return pd.DataFrame(columns=["location", "s3_file", "size", "created", "modified"])

    def check(self, location: str) -> bool:
        """Check if a DataFrame exists at the specified location

        Args:
            location (str): The location of the data to check.

        Returns:
            bool: True if the data exists, False otherwise.
        """
        # Generate the specific S3 prefix for the target location
        s3_prefix = f"{self.path_prefix}/{location}.parquet/"
        s3_prefix = re.sub(r"/+", "/", s3_prefix)  # Collapse slashes

        # Use list_objects_v2 to check if any objects exist under this specific prefix
        response = self.s3_client.list_objects_v2(Bucket=self.workbench_bucket, Prefix=s3_prefix, MaxKeys=1)
        return "Contents" in response

    @not_found_returns_none
    def get(self, location: str) -> Union[pd.DataFrame, None]:
        """Retrieve a DataFrame from AWS S3.

        Args:
            location (str): The location of the data to retrieve.

        Returns:
            pd.DataFrame: The retrieved DataFrame or None if not found.
        """
        s3_uri = self._generate_s3_uri(location)
        return wr.s3.read_parquet(s3_uri)

    def upsert(self, location: str, data: Union[pd.DataFrame, pd.Series]):
        """Insert or update a DataFrame or Series in the AWS S3.

        Args:
            location (str): The location of the data.
            data (Union[pd.DataFrame, pd.Series]): The data to be stored.
        """
        # Check if the data is a Pandas Series, convert it to a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only Pandas DataFrame or Series objects are supported.")

        # Convert object columns to string type to avoid PyArrow type inference issues.
        data = self.type_convert_before_parquet(data)

        # Update/Insert the DataFrame to S3
        s3_uri = self._generate_s3_uri(location)
        try:
            wr.s3.to_parquet(df=data, path=s3_uri, dataset=True, mode="overwrite", index=True)
            self.log.info(f"Dataframe cached {s3_uri}...")
        except Exception as e:
            self.log.error(f"Failed to cache dataframe '{s3_uri}': {e}")
            raise

    def append(self, location: str, data: Union[pd.DataFrame, pd.Series]):
        """Append a DataFrame as a new unique-named parquet file under the location.

        Unlike :meth:`upsert`, this does not delete existing files, so it is
        safe for concurrent writers: each caller lands its own file under the
        dataset prefix. Readers (``get``) transparently read all files as
        one dataset. Schemas across files must be compatible — callers that
        produce drifting dtypes are responsible for normalizing before append.

        Args:
            location (str): The location of the data.
            data (Union[pd.DataFrame, pd.Series]): The data to be appended.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only Pandas DataFrame or Series objects are supported.")

        data = self.type_convert_before_parquet(data)

        s3_uri = self._generate_s3_uri(location)
        part_prefix = f"part-{uuid.uuid4().hex}-"
        try:
            wr.s3.to_parquet(
                df=data,
                path=s3_uri,
                dataset=True,
                mode="append",
                filename_prefix=part_prefix,
                index=True,
            )
            self.log.info(f"Dataframe appended to {s3_uri} ({part_prefix})...")
        except Exception as e:
            self.log.error(f"Failed to append dataframe '{s3_uri}': {e}")
            raise

    @staticmethod
    def type_convert_before_parquet(df: pd.DataFrame) -> pd.DataFrame:
        # Convert object columns to string type to avoid PyArrow type inference issues.
        df = df.copy()
        object_cols = df.select_dtypes(include=["object"]).columns
        df[object_cols] = df[object_cols].astype("str")
        return df

    def delete(self, location: str):
        """Delete a DataFrame from the AWS S3.

        Args:
            location (str): The location of the data to delete.
        """
        s3_uri = self._generate_s3_uri(location)

        # Check if the folder (prefix) exists in S3
        if not wr.s3.list_objects(s3_uri):
            self.log.info(f"Data '{location}' does not exist in S3...")
            return

        # Delete the data from S3
        try:
            wr.s3.delete_objects(s3_uri)
            self.log.info(f"Data '{location}' deleted successfully from S3.")
        except Exception as e:
            self.log.error(f"Failed to delete data '{location}': {e}")

    def delete_recursive(self, location: str):
        """Recursively delete all data under the specified location in AWS S3.

        Args:
            location (str): The location prefix of the data to delete.
        """
        # Construct the full prefix for S3
        s3_prefix = re.sub(r"/+", "/", f"{self.path_prefix}/{location}")  # Collapse slashes
        s3_prefix = s3_prefix.rstrip("/") + "/"  # Ensure the prefix ends with a slash

        # List all objects under the given prefix
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.workbench_bucket, Prefix=s3_prefix)
            if "Contents" not in response:
                self.log.info(f"No data found under '{s3_prefix}' to delete.")
                return

            # Gather all keys to delete
            keys = [{"Key": obj["Key"]} for obj in response["Contents"]]
            response = self.s3_client.delete_objects(Bucket=self.workbench_bucket, Delete={"Objects": keys})
            for response in response.get("Deleted", []):
                self.log.info(f"Deleted: {response['Key']}")

        except Exception as e:
            self.log.error(f"Failed to delete data recursively at '{location}': {e}")

    def _generate_s3_uri(self, location: str) -> str:
        """Generate the S3 URI for the given location."""
        s3_path = f"{self.workbench_bucket}/{self.path_prefix}/{location}.parquet"
        return f"s3://{re.sub(r'/+', '/', s3_path)}"

    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse an S3 URI into bucket and key."""
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    def __repr__(self):
        """Return a string representation of the DFStore object."""
        # Use the summary() method and format it to align columns for printing
        summary_df = self.summary()

        # Sanity check: If there are no objects, return a message
        if summary_df.empty:
            return "DFStoreCore: No data objects found in the store."

        # Dynamically compute the max length of the 'location' column and add 5 spaces for padding
        max_location_len = summary_df["location"].str.len().max() + 2
        summary_df["location"] = summary_df["location"].str.ljust(max_location_len)

        # Format the size column to include (MB) and ensure 3 spaces between size and date
        summary_df["size (MB)"] = summary_df["size (MB)"].apply(lambda x: f"{x:.2f} MB")

        # Enclose the modified date in parentheses and ensure 3 spaces between size and date
        summary_df["modified"] = summary_df["modified"].apply(lambda x: f" ({x})")

        # Convert the DataFrame to a string, remove headers, and return
        return summary_df.to_string(index=False, header=False)
