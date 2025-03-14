"""SQL based Sample Rows: Compute Sample rows for a DataSource using SQL"""

import logging
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.data_source_abstract import DataSourceAbstract

# Workbench Logger
log = logging.getLogger("workbench")


def sample_rows(data_source: DataSourceAbstract) -> pd.DataFrame:
    """Pull a sample of rows from the DataSource
    Args:
        data_source: The DataSource that we're pulling the sample rows from
    Returns:
        pd.DataFrame: A sample DataFrame from this DataSource
    """

    # Grab the DataSource computation table name
    table = data_source.view("computation").table

    # Get the column names the DataSource computation view
    column_names = data_source.view("computation").columns
    sql_columns = ", ".join([f'"{name}"' for name in column_names])

    # Note: Hardcoded to 100 rows so that metadata storage is consistent
    sample_rows = 100
    num_rows = data_source.num_rows()
    if num_rows > sample_rows:
        # Bernoulli Sampling has reasonable variance, so we're going to +1 the
        # sample percentage and then simply clamp it to 100 rows
        percentage = round(sample_rows * 100.0 / num_rows) + 1
        data_source.log.info(f"DataSource has {num_rows} rows.. sampling down to {sample_rows}...")
        query = f'SELECT {sql_columns} FROM "{table}" TABLESAMPLE BERNOULLI({percentage})'
    else:
        query = f'SELECT {sql_columns} FROM "{table}"'
    sample_df = data_source.query(query)

    # Sanity Check
    if sample_df is None:
        log.error(f"Error pulling sample rows from {data_source.uuid}")
        return None

    # Grab the first 100 rows
    sample_df = sample_df.head(sample_rows)

    # Return the sample_df
    return sample_df


if __name__ == "__main__":
    """Exercise the SQL Sample Rows Functionality"""
    from workbench.api.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("test_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

    # What's my Workbench UUID
    print(f"UUID: {my_data.uuid}")

    # Get sample rows for this DataSource
    my_sample_df = sample_rows(my_data)
    print("\nSample Rows")
    print(my_sample_df)
