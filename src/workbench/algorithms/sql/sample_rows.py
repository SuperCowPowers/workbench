"""SQL based Sample Rows: Compute Sample rows for a DataSource using SQL"""

import logging
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.data_source_abstract import DataSourceAbstract

# Workbench Logger
log = logging.getLogger("workbench")


def sample_rows(data_source: DataSourceAbstract, rows: int = 100) -> pd.DataFrame:
    """Pull a sample of rows from the DataSource

    Args:
        data_source: The DataSource that we're pulling the sample rows from
        rows: The number of rows to sample from the DataSource (default: 100)

    Returns:
        pd.DataFrame: A sample DataFrame from this DataSource
    """

    # Grab the DataSource computation table name
    table = data_source.view("computation").table

    # Get the column names the DataSource computation view
    column_names = data_source.view("computation").columns
    sql_columns = ", ".join([f'"{name}"' for name in column_names])

    # Downsample the DataSource based on Bernoulli Sampling
    num_rows = data_source.num_rows()
    if num_rows > rows:
        # Bernoulli Sampling has reasonable variance, so we're going to add a 10% fudge factor
        # to the sample percentage and then simply clamp it to desired rows
        percentage = round(rows * 110.0 / num_rows)
        data_source.log.info(f"DataSource has {num_rows} rows.. sampling down to {rows}...")
        query = f'SELECT {sql_columns} FROM "{table}" TABLESAMPLE BERNOULLI({percentage})'
    else:
        query = f'SELECT {sql_columns} FROM "{table}"'
    sample_df = data_source.query(query)

    # Sanity Check
    if sample_df is None:
        log.error(f"Error pulling sample rows from {data_source.name}")
        return None

    # Grab the first N rows (this clamps the sample_df to the desired number of rows)
    sample_df = sample_df.head(rows)

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
    my_data = DataSource("abalone_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

    # What's my Workbench Name
    print(f"Name: {my_data.name}")

    # Sample rows for this DataSource
    my_sample_df = sample_rows(my_data)
    print("\nSample Rows")
    print(my_sample_df)

    # Get a larger sample of rows for this DataSource
    my_sample_df = sample_rows(my_data, rows=1000)
    print("\nSample Rows")
    print(my_sample_df)
