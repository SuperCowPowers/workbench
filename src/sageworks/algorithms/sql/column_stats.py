"""SQL based Column Statistics: Compute Column Statistics for a DataSource using SQL"""
import logging
import operator
import pandas as pd

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


def column_stats(data_source: DataSourceAbstract) -> dict[dict]:
    """SQL based Column Statistics: Compute Column Statistics for a DataSource using SQL
    Args:
        data_source(DataSource): The DataSource that we're computing column stats on
    Returns:
        dict(dict): A dictionary of stats for each column this format
        NB: String columns will have value_counts but NOT have num_zeros and quartiles
             {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12, 'value_counts': {...}},
              'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100, 'quartiles': {...}},
              ...}
    """

    # The DataSource class will have column names and types
    data_source.log.info("Computing Column Statistics for all columns...")

    # Get the column names and types from the DataSource
    columns_stats = {name: {"dtype": dtype} for name, dtype in data_source.column_details().items()}

    # Now add quartiles to the column stats
    quartiles = data_source.quartiles()
    for column, quartile_info in quartiles.items():
        columns_stats[column]["quartiles"] = quartile_info

    # Now add value_counts to the column stats
    value_counts = data_source.value_counts()
    for column, count_info in value_counts.items():
        columns_stats[column]["value_counts"] = count_info

    # For every column in the table get unique values and Nulls/NaNs
    # Also for numeric columns get the number of zero values
    data_source.log.info("Computing Unique values and num zero for numeric columns (this may take a while)...")
    numeric = ["tinyint", "smallint", "int", "bigint", "float", "double", "decimal"]
    for column, data_type in data_source.column_details().items():
        log.info(f"Computing columns_stats for column: {column}...")

        # Compute number of unique values
        num_unique_query = f'SELECT COUNT(DISTINCT "{column}") AS unique_values FROM {data_source.table_name}'
        num_unique = data_source.query(num_unique_query).iloc[0]["unique_values"]
        columns_stats[column]["unique"] = num_unique

        # Compute number of nulls
        num_nulls_query = f'SELECT COUNT(*) AS num_nulls FROM {data_source.table_name} WHERE "{column}" IS NULL'
        num_nulls = data_source.query(num_nulls_query).iloc[0]["num_nulls"]
        columns_stats[column]["nulls"] = num_nulls

        # If numeric, compute number of zeros
        if data_type in numeric:
            query = f'SELECT COUNT(*) AS num_zeros FROM {data_source.table_name} WHERE "{column}" = 0'
            num_zeros = data_source.query(query).iloc[0]["num_zeros"]
            columns_stats[column]["num_zeros"] = num_zeros

    # Return the quartile data
    return columns_stats


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
    assert my_data.check()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # Get column stats for all columns
    my_column_stats = column_stats(my_data)
    print("\nColumn Stats:")
    pprint(my_column_stats)
