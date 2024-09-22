"""DFStore: Manages SageWorks data in AWS S3 using Pandas DataFrames and AWS Wrangler."""

from typing import Union
import logging
import awswrangler as wr
import pandas as pd
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager


class DFStore:
    """DFStore: Manages SageWorks data in AWS S3 using Pandas DataFrames.

    Common Usage:
        ```
        df_store = DFStore()

        # List Data
        df_store.list()

        # Add DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df_store.add("my_data", df)

        # Retrieve DataFrame
        df = df_store.get("my_data")
        print(df)

        # Delete Data
        df_store.delete("my_data")
        ```
    """

    def __init__(self):
        """DFStore Init Method"""
        self.log = logging.getLogger("sageworks")
        self.prefix = "df_store/"

        # Initialize a SageWorks Session and retrieve the S3 bucket from ConfigManager
        config = ConfigManager()
        self.sageworks_bucket = config.get_config("SAGEWORKS_BUCKET")

        # Grab a SageWorks Session (this allows us to assume the SageWorks ExecutionRole)
        self.boto3_session = AWSAccountClamp().boto3_session

        # Read all the Pipelines from this S3 path
        self.s3_client = self.boto3_session.client("s3")

    def list(self) -> list:
        """Return a nicely formatted list of object names, sizes (in MB), and modified date."""
        try:
            df = self.details()
            formatted_list = [
                f"{row['name']} ({row['size'] / (1024 * 1024):.3f} MB) Last Modified: {row['modified'].strftime('%Y-%m-%d %H:%M:%S')}"
                for _, row in df.iterrows()
            ]
            return formatted_list
        except Exception as e:
            self.log.error(f"Failed to list objects: {e}")
            return []

    def details(self) -> pd.DataFrame:
        """Return a DataFrame with detailed metadata for all objects in the data_store prefix."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.sageworks_bucket, Prefix=self.prefix)

            if 'Contents' not in response:
                return pd.DataFrame(columns=['name', 's3_file', 'size', 'created', 'modified'])

            # Collect details for each object
            data = []
            for obj in response['Contents']:
                name = obj["Key"].replace(f"{self.prefix}", "")
                s3_file = f"s3://{self.sageworks_bucket}/{obj['Key']}"
                size = obj["Size"]
                created = obj["LastModified"]
                modified = obj["LastModified"]
                data.append([name, s3_file, size, created, modified])

            # Create and return DataFrame
            df = pd.DataFrame(data, columns=['name', 's3_file', 'size', 'created', 'modified'])
            return df

        except Exception as e:
            self.log.error(f"Failed to get object details: {e}")
            return pd.DataFrame(columns=['name', 's3_file', 'size', 'created', 'modified'])

    def get(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame from the AWS S3.

        Args:
            name (str): The name of the data to retrieve.

        Returns:
            pd.DataFrame: The retrieved DataFrame.
        """
        s3_uri = self._generate_s3_uri(name)
        try:
            df = wr.s3.read_parquet(s3_uri)
            return df
        except ClientError as e:
            self.log.warning(f"Data '{name}' not found in S3.")
            return pd.DataFrame()  # Return an empty DataFrame if not found
        except Exception as e:
            self.log.error(f"Failed to retrieve data '{name}': {e}")
            return pd.DataFrame()

    def add(self, name: str, data: Union[pd.DataFrame, pd.Series], overwrite: bool = True):
        """Add or update a DataFrame or Series in the AWS S3.

        Args:
            name (str): The name of the data.
            data (Union[pd.DataFrame, pd.Series]): The data to be stored.
            overwrite (bool): Whether to overwrite existing data.
        """
        # Check if the data is a Pandas Series, convert it to a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only Pandas DataFrame or Series objects are supported.")

        s3_uri = self._generate_s3_uri(name)
        try:
            # Use 'overwrite' mode if overwrite is True, otherwise check for existence and raise an error
            mode = 'overwrite' if overwrite else None
            wr.s3.to_parquet(df=data, path=s3_uri, dataset=True, mode=mode)
            self.log.info(f"Data '{name}' added/updated successfully in S3.")
        except Exception as e:
            if not overwrite and wr.s3.does_object_exist(s3_uri):
                self.log.error(f"Data '{name}' already exists in S3 and overwrite is set to False.")
                raise FileExistsError(f"Data '{name}' already exists in S3.")
            else:
                self.log.critical(f"Failed to add/update data '{name}': {e}")
                raise

    def delete(self, name: str):
        """Delete a DataFrame from the AWS S3.

        Args:
            name (str): The name of the data to delete.
        """
        s3_uri = self._generate_s3_uri(name)
        try:
            wr.s3.delete_objects(s3_uri)
            self.log.info(f"Data '{name}' deleted successfully from S3.")
        except Exception as e:
            self.log.error(f"Failed to delete data '{name}': {e}")

    def _generate_s3_uri(self, name: str) -> str:
        """Generate the S3 URI for the given name."""
        s3_path = f"{self.sageworks_bucket}/{self.prefix}{name}.parquet"
        s3_path = s3_path.replace("//", "/")
        s3_uri = f"s3://{s3_path}"
        return s3_uri

    def __repr__(self):
        """Return a string representation of the DFStore object."""
        return "\n".join(self.list())


if __name__ == "__main__":
    """Exercise the DFStore Class"""

    # Create a DFStore manager
    df_store = DFStore()

    # List the data
    print("Listing Data...")
    print(df_store.list())

    # Add a new DataFrame
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_store.add("test_data", df, overwrite=True)

    # Get the DataFrame
    print(f"Getting data 'test_data':\n{df_store.get('test_data')}")

    # Delete the data
    df_store.delete("test_data")

    # Now let's test adding a Series
    series = pd.Series([1, 2, 3, 4], name="Series")
    df_store.add("test_series", series, overwrite=True)
    print(f"Getting data 'test_series':\n{df_store.get('test_series')}")
    df_store.delete("test_series")