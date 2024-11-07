"""AWSDFStore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy"""

from typing import Union
import logging
import awswrangler as wr
import pandas as pd
import re

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.aws_utils import not_found_returns_none


class AWSDFStore:
    """AWSDFStore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy

    Common Usage:
        ```python
        df_store = AWSDFStore()

        # List Data
        df_store.list()

        # Add DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df_store.upsert("my_data", df)

        # Retrieve DataFrame
        df = df_store.get("my_data")
        print(df)

        # Delete Data
        df_store.delete("my_data")
        ```
    """

    def __init__(self, path_prefix: Union[str, None] = None):
        """AWSDFStore Init Method

        Args:
            path_prefix (Union[str, None], optional): Add a path prefix to storage locations (Defaults to None)
        """
        self.log = logging.getLogger("sageworks")
        self._base_prefix = "df_store/"
        self.path_prefix = self._base_prefix + path_prefix if path_prefix else self._base_prefix
        self.path_prefix = re.sub(r"/+", "/", self.path_prefix)  # Collapse slashes

        # Initialize a SageWorks Session and retrieve the S3 bucket from ConfigManager
        config = ConfigManager()
        self.sageworks_bucket = config.get_config("SAGEWORKS_BUCKET")

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session

        # Read all the Pipelines from this S3 path
        self.s3_client = self.boto3_session.client("s3")

    def summary(self) -> pd.DataFrame:
        """Return a nicely formatted summary of object locations, sizes (in MB), and modified dates."""
        df = self.details()

        # Create a formatted DataFrame
        formatted_df = pd.DataFrame(
            {
                "location": df["location"],
                "size (MB)": (df["size"] / (1024 * 1024)).round(2),  # Convert size to MB
                "modified": pd.to_datetime(df["modified"]).dt.strftime("%Y-%m-%d %H:%M:%S"),  # Format date
            }
        )
        return formatted_df

    def details(self) -> pd.DataFrame:
        """Return a DataFrame with detailed metadata for all objects in the data_store prefix."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.sageworks_bucket, Prefix=self.path_prefix)
            if "Contents" not in response:
                return pd.DataFrame(columns=["location", "s3_file", "size", "modified"])

            # Collect details for each object
            data = []
            for obj in response["Contents"]:
                full_key = obj["Key"]

                # Reverse logic: Strip the bucket/prefix in the front and .parquet in the end
                location = full_key.replace(f"{self.path_prefix}", "/").split(".parquet")[0]
                s3_file = f"s3://{self.sageworks_bucket}/{full_key}"
                size = obj["Size"]
                modified = obj["LastModified"]
                data.append([location, s3_file, size, modified])

            # Create and return DataFrame
            df = pd.DataFrame(data, columns=["location", "s3_file", "size", "modified"])
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
        response = self.s3_client.list_objects_v2(Bucket=self.sageworks_bucket, Prefix=s3_prefix, MaxKeys=1)
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

        s3_uri = self._generate_s3_uri(location)
        try:
            wr.s3.to_parquet(df=data, path=s3_uri, dataset=True, mode="overwrite")
            self.log.info(f"Dataframe cached {s3_uri}...")
        except Exception as e:
            self.log.error(f"Failed to cache dataframe '{s3_uri}': {e}")
            raise

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

        # List all objects under the given prefix
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.sageworks_bucket, Prefix=s3_prefix)
            if "Contents" not in response:
                self.log.info(f"No data found under '{s3_prefix}' to delete.")
                return

            # Gather all keys to delete
            keys = [{"Key": obj["Key"]} for obj in response["Contents"]]
            response = self.s3_client.delete_objects(Bucket=self.sageworks_bucket, Delete={"Objects": keys})
            for response in response.get("Deleted", []):
                self.log.info(f"Deleted: {response['Key']}")

        except Exception as e:
            self.log.error(f"Failed to delete data recursively at '{location}': {e}")

    def _generate_s3_uri(self, location: str) -> str:
        """Generate the S3 URI for the given location."""
        s3_path = f"{self.sageworks_bucket}/{self.path_prefix}/{location}.parquet"
        s3_path = re.sub(r"/+", "/", s3_path)  # Collapse slashes
        s3_uri = f"s3://{s3_path}"
        return s3_uri

    def __repr__(self):
        """Return a string representation of the AWSDFStore object."""
        # Use the summary() method and format it to align columns for printing
        summary_df = self.summary()

        # Sanity check: If there are no objects, return a message
        if summary_df.empty:
            return "AWSDFStore: No data objects found in the store."

        # Dynamically compute the max length of the 'location' column and add 5 spaces for padding
        max_location_len = summary_df["location"].str.len().max() + 2
        summary_df["location"] = summary_df["location"].str.ljust(max_location_len)

        # Format the size column to include (MB) and ensure 3 spaces between size and date
        summary_df["size (MB)"] = summary_df["size (MB)"].apply(lambda x: f"{x:.2f} MB")

        # Enclose the modified date in parentheses and ensure 3 spaces between size and date
        summary_df["modified"] = summary_df["modified"].apply(lambda x: f" ({x})")

        # Convert the DataFrame to a string, remove headers, and return
        return summary_df.to_string(index=False, header=False)


if __name__ == "__main__":
    """Exercise the AWSDFStore Class"""
    import time

    # Create a AWSDFStore manager
    df_store = AWSDFStore()

    # Details of the Dataframe Store
    print("Detailed Data...")
    print(df_store.details())

    # Add a new DataFrame
    my_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_store.upsert("/testing/test_data", my_df)

    # Get the DataFrame
    print(f"Getting data 'test_data':\n{df_store.get('/testing/test_data')}")

    # Now let's test adding a Series
    series = pd.Series([1, 2, 3, 4], name="Series")
    df_store.upsert("/testing/test_series", series)
    print(f"Getting data 'test_series':\n{df_store.get('/testing/test_series')}")

    # Summary of the data
    print("Summary Data...")
    print(df_store.summary())

    # Repr of the AWSDFStore object
    print("AWSDFStore Object:")
    print(df_store)

    # Check if the data exists
    print("Check if data exists...")
    print(df_store.check("/testing/test_data"))
    print(df_store.check("/testing/test_series"))

    # Time the check
    start_time = time.time()
    print(df_store.check("/testing/test_data"))
    print("--- Check %s seconds ---" % (time.time() - start_time))

    # Now delete the test data
    df_store.delete("/testing/test_data")
    df_store.delete("/testing/test_series")

    # Check if the data exists
    print("Check if data exists...")
    print(df_store.check("/testing/test_data"))
    print(df_store.check("/testing/test_series"))

    # Add a bunch of dataframes and then test recursive delete
    for i in range(10):
        df_store.upsert(f"/testing/data_{i}", pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
    print("Before Recursive Delete:")
    print(df_store.summary())
    df_store.delete_recursive("/testing")
    print("After Recursive Delete:")
    print(df_store.summary())

    # Get a non-existent DataFrame
    print("Getting non-existent data...")
    print(df_store.get("/testing/no_where"))

    # Test path_prefix
    df_store = AWSDFStore(path_prefix="/super/test")
    print(df_store.path_prefix)
    df_store.upsert("test_data", my_df)
    print(df_store.get("test_data"))
    print(df_store.summary())
    df_store.delete("test_data")
    print(df_store.summary())
