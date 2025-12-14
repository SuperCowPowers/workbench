"""DFStoreCore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy"""

import logging
from typing import Union

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Workbench Bridges Import
from workbench_bridges.api import DFStore as BridgesDFStore


class DFStoreCore(BridgesDFStore):
    """DFStoreCore: Fast/efficient storage of DataFrames using AWS S3/Parquet/Snappy

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

    def __init__(self, path_prefix: Union[str, None] = None):
        """DFStoreCore Init Method

        Args:
            path_prefix (Union[str, None], optional): Add a path prefix to storage locations (Defaults to None)
        """
        # Get config from workbench's systems
        bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
        session = AWSAccountClamp().boto3_session

        # Initialize parent with workbench config
        super().__init__(path_prefix=path_prefix, s3_bucket=bucket, boto3_session=session)
        self.log = logging.getLogger("workbench")


if __name__ == "__main__":
    """Exercise the DFStoreCore Class"""
    import time
    import pandas as pd

    # Create a DFStoreCore manager
    df_store = DFStoreCore()

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

    # Repr of the DFStoreCore object
    print("DFStoreCore Object:")
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
