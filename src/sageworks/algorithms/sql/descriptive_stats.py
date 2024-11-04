"""SQL based Descriptive Stats: Compute Descriptive Stats for all the numeric columns in a DataSource using SQL"""

import logging
import pandas as pd
from collections import defaultdict

# SageWorks Imports
from sageworks.core.artifacts.data_source_abstract import DataSourceAbstract


# SageWorks Logger
log = logging.getLogger("sageworks")


def descriptive_stats_query(columns: list[str], table_name: str) -> str:
    """Build a query to compute the descriptive stats for all columns in a table
    Args:
        columns(list(str)): The columns to compute descriptive stats on
        table_name(str): The table to compute descriptive stats on
    Returns:
        str: The SQL query to compute descriptive stats
    """
    query = f'SELECT <<column_descriptive_stats>> FROM "{table_name}"'
    column_descriptive_stats = ""
    for c in columns:
        column_descriptive_stats += (
            f'min("{c}") AS "{c}___min", '
            f'approx_percentile("{c}", 0.25) AS "{c}___q1", '
            f'approx_percentile("{c}", 0.5) AS "{c}___median", '
            f'approx_percentile("{c}", 0.75) AS "{c}___q3", '
            f'max("{c}") AS "{c}___max", '
            f'avg("{c}") AS "{c}___mean", '
            f'stddev("{c}") AS "{c}___stddev", '
        )
    query = query.replace("<<column_descriptive_stats>>", column_descriptive_stats[:-2])

    # Return the query
    return query


def descriptive_stats(data_source: DataSourceAbstract) -> dict[dict]:
    """Compute Descriptive Stats for all the numeric columns in a DataSource
    Args:
        data_source(DataSource): The DataSource that we're computing descriptive stats on
    Returns:
        dict(dict): A dictionary of descriptive stats for each column in this format
             {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4, 'mean': 2.5, 'stddev': 1.5},
              'col2': ...}
    """
    # Grab the DataSource computation view table name
    table = data_source.view("computation").table

    # Figure out which columns are numeric
    num_type = ["double", "float", "int", "bigint", "smallint", "tinyint"]
    details = data_source.view("computation").column_details()
    numeric = [column for column, data_type in details.items() if data_type in num_type]

    # Sanity Check for numeric columns
    if len(numeric) == 0:
        log.warning("No numeric columns found in the current computation view of the DataSource")
        log.warning("If the data source was created from a DataFrame, ensure that the DataFrame was properly typed")
        log.warning("Recommendation: Properly type the DataFrame and recreate the SageWorks artifact")
        return {}

    # Build the query
    query = descriptive_stats_query(numeric, table)

    # Run the query
    log.debug(query)
    result_df = data_source.query(query)

    # Process the results
    # Note: The result_df is a DataFrame with a single row and a column for each stat metric
    stats_dict = result_df.to_dict(orient="index")[0]

    # Convert the dictionary to a nested dictionary
    # Note: The keys are in the format col1__col2
    nested_descriptive_stats = defaultdict(dict)
    for key, value in stats_dict.items():
        col1, col2 = key.split("___")
        nested_descriptive_stats[col1][col2] = value

    # Return the nested dictionary
    return dict(nested_descriptive_stats)


if __name__ == "__main__":
    """Exercise the SQL Descriptive Stats Functionality"""
    from pprint import pprint
    from sageworks.api.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("test_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # Get descriptive_stats for numeric columns
    my_descriptive_stats = descriptive_stats(my_data)
    print("\nDescriptive Stats")
    pprint(my_descriptive_stats)
