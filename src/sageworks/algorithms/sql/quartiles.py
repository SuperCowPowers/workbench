"""SQL based Quartiles: Compute Quartiles for all the numeric columns in a DataSource using SQL"""
import logging
import pandas as pd
from collections import defaultdict

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


def quartiles_query(columns: list[str], table_name: str) -> str:
    """Build a query to compute the quartiles for all columns in a table
    Args:
        columns(list(str)): The columns to compute quartiles on
        table_name(str): The table to compute quartiles on
    Returns:
        str: The SQL query to compute quartiles
    """
    query = f"SELECT <<column_quartiles>> FROM {table_name}"
    column_quartiles = ""
    for c in columns:
        column_quartiles += (
            f'min("{c}") AS {c}__min, '
            f'approx_percentile("{c}", 0.25) AS {c}__q1, '
            f'approx_percentile("{c}", 0.5) AS {c}__median, '
            f'approx_percentile("{c}", 0.75) AS {c}__q3, '
            f'max("{c}") AS {c}__max, '
        )
    query = query.replace("<<column_quartiles>>", column_quartiles[:-2])

    # Return the query
    return query


def quartiles(data_source: DataSourceAbstract) -> dict[dict]:
    """Compute Quartiles for all the numeric columns in a DataSource
    Args:
        data_source(DataSource): The DataSource that we're computing quartiles on
    Returns:
        dict(dict): A dictionary of quartiles for each column in this format
             {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
              'col2': ...}
    """

    # Figure out which columns are numeric
    num_type = ["double", "float", "int", "bigint", "smallint", "tinyint"]
    details = data_source.column_details()
    numeric = [column for column, data_type in details.items() if data_type in num_type]

    # Build the query
    query = quartiles_query(numeric, data_source.table_name)

    # Run the query
    result_df = data_source.query(query)

    # Process the results
    # Note: The result_df is a DataFrame with a single row and a column for each quartile metric
    quartile_dict = result_df.to_dict(orient="index")[0]

    # Convert the dictionary to a nested dictionary
    # Note: The keys are in the format col1__col2
    nested_quartiles = defaultdict(dict)
    for key, value in quartile_dict.items():
        col1, col2 = key.split("__")
        nested_quartiles[col1][col2] = value

    # Return the nested dictionary
    return dict(nested_quartiles)


if __name__ == "__main__":
    """Exercise the SQL Quartiles Functionality"""
    from pprint import pprint
    from sageworks.artifacts.data_sources.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("abalone_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # Get quartiles for numeric columns
    my_quartiles = quartiles(my_data)
    print("\nQuartiles")
    pprint(my_quartiles)
