"""DFStore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy"""

from typing import Union
import logging
import pandas as pd

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_df_store import AWSDFStore


class DFStore(AWSDFStore):
    """DFStore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy

    Common Usage:
        ```python
        df_store = DFStore()

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

    def __init__(self, path_prefix: Union[str, None] = None):
        """DFStore Init Method

        Args:
            path_prefix (Union[str, None], optional): Add a path prefix to storage locations (Defaults to None)
        """
        self.log = logging.getLogger("sageworks")

        # Initialize the SuperClass
        super().__init__(path_prefix=path_prefix)

    def summary(self) -> pd.DataFrame:
        """Return a nicely formatted summary of object locations, sizes (in MB), and modified dates.

        Returns:
            pd.DataFrame: A formatted DataFrame with the summary details.
        """
        return super().summary()

    def details(self) -> pd.DataFrame:
        """Return a DataFrame with detailed metadata for all objects in the data_store prefix.

        Returns:
            pd.DataFrame: A DataFrame with detailed metadata for all objects in the data_store prefix.
        """
        return super().details()

    def check(self, location: str) -> bool:
        """Check if a DataFrame exists at the specified location

        Args:
            location (str): The location of the data to check.

        Returns:
            bool: True if the data exists, False otherwise.
        """
        return super().check(location)

    def get(self, location: str) -> Union[pd.DataFrame, None]:
        """Retrieve a DataFrame from AWS S3.

        Args:
            location (str): The location of the data to retrieve.

        Returns:
            pd.DataFrame: The retrieved DataFrame or None if not found.
        """
        return super().get(location)

    def upsert(self, location: str, data: Union[pd.DataFrame, pd.Series]):
        """Insert or update a DataFrame or Series in the AWS S3.

        Args:
            location (str): The location of the data.
            data (Union[pd.DataFrame, pd.Series]): The data to be stored.
        """
        super().upsert(location, data)

    def delete(self, location: str):
        """Delete a DataFrame from the AWS S3.

        Args:
            location (str): The location of the data to delete.
        """
        super().delete(location)


if __name__ == "__main__":
    """Exercise the DFStore Class"""
    import time

    # Create a DFStore manager
    df_store = DFStore()

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

    # Repr of the DFStore object
    print("DFStore Object:")
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
