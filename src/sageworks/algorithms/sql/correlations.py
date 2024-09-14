"""SQL based Correlations: Compute Correlations for all the numeric columns in a DataSource using SQL"""

import logging
import pandas as pd
from collections import defaultdict

# SageWorks Imports
from sageworks.core.artifacts.data_source_abstract import DataSourceAbstract


# SageWorks Logger
log = logging.getLogger("sageworks")


def correlation_query(columns: list[str], table_name: str) -> str:
    """Build a query to compute the correlations between columns in a table
    Args:
        columns(list(str)): The columns to compute correlations on
        table_name(str): The table to compute correlations on
    Returns:
        str: The SQL query to compute correlations
    """
    query = f"SELECT <<cross_correlations>> FROM {table_name}"
    cross_correlations = ""
    for c in columns:
        for d in columns:
            if c != d:
                cross_correlations += f'corr("{c}", "{d}") AS "{c}__{d}", '
    query = query.replace("<<cross_correlations>>", cross_correlations[:-2])

    # Return the query
    return query


def correlations(data_source: DataSourceAbstract) -> dict[dict]:
    """Compute Correlations for all the numeric columns in a DataSource
    Args:
        data_source(DataSource): The DataSource that we're computing correlations on
    Returns:
        dict(dict): A dictionary of correlations for each column in this format
             {'col1': {'col2': 0.5, 'col3': 0.9, 'col4': 0.4, ...},
              'col2': {'col1': 0.5, 'col3': 0.8, 'col4': 0.3, ...}}
    """
    data_source.log.info("Computing Correlations for numeric columns...")

    # Figure out which columns are numeric
    num_type = ["double", "float", "int", "bigint", "smallint", "tinyint"]
    details = data_source.view("computation").column_details()

    # Get the numeric columns
    numeric = [column for column, data_type in details.items() if data_type in num_type]

    # If we have at least two numeric columns, compute the correlations
    if len(numeric) < 2:
        return {}

    # Grab the DataSource computation table name
    table = data_source.view("computation").table

    # Build the query
    query = correlation_query(numeric, table)

    # Run the query
    log.debug(query)
    result_df = data_source.query(query)

    # Drop any columns that have NaNs
    result_df = result_df.dropna(axis=1)

    # Process the results
    # Note: The result_df is a DataFrame with a single row and a column for each pairwise correlation
    correlation_dict = result_df.to_dict(orient="index")[0]

    # Convert the dictionary to a nested dictionary
    # Note: The keys are in the format col1__col2
    nested_corr = defaultdict(dict)
    for key, value in correlation_dict.items():
        col1, col2 = key.split("__")
        nested_corr[col1][col2] = value

    # Sort the nested dictionaries
    sorted_dict = {}
    for key, sub_dict in nested_corr.items():
        sorted_dict[key] = {k: v for k, v in sorted(sub_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict


if __name__ == "__main__":
    """Exercise the SQL Correlations Functionality"""
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

    # Get correlations for numeric columns
    my_correlations = correlations(my_data)
    print("\nCorrelations")
    print(my_correlations)
