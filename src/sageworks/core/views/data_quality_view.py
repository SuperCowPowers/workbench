"""Create a Data Quality View: A View that computes various data_quality metrics"""

import logging
from typing import Union
import numpy as np

# SageWorks Imports
from sageworks.api import DataSource
from sageworks.core.views.view_utils import dataframe_to_table, get_column_list

log = logging.getLogger("sageworks")


def create_data_quality_view(data_source: DataSource, id_column: str, source_table: str = None):
    """Create a Data Quality View: A View that computes various data_quality metrics

    Args:
        data_source (DataSource): The DataSource object
        id_column (str): The name of the id column (must be defined for join logic)
        source_table_name (str, optional): The table/view to create the view from. Defaults to data_source base table.
    """

    # Set the source_table to create the view from
    base_table = data_source.get_table_name()
    source_table = source_table if source_table else base_table

    # Check the number of rows in the source_table, if greater than 1M, then give an error and return
    row_count = data_source.num_rows()
    if row_count > 1_000_000:
        log.error(f"Data Quality View cannot be created on more than 1M rows. {source_table} has {row_count} rows.")
        return

    # Drop any columns generated from AWS
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    source_table_columns = get_column_list(data_source, source_table)
    column_list = [col for col in source_table_columns if col not in aws_cols]

    # Enclose each column name in double quotes
    sql_columns = ", ".join([f'"{column}"' for column in column_list])

    # Pull in the data from the source_table
    query = f"SELECT {sql_columns} FROM {source_table}"
    df = data_source.query(query)

    # Check if the id_column exists in the source_table
    if id_column not in df.columns:
        log.error(f"id_column {id_column} not found in {source_table}. Cannot create Data Quality View.")
        return

    # TEMP: Just make a random data_quality column with random float from 0-1
    df["data_quality"] = np.random.rand(row_count)

    # Create the data_quality supplemental table
    data_quality_table = f"_{base_table}_data_quality"
    dataframe_to_table(data_source, df[[id_column, "data_quality"]], data_quality_table)

    # Create the data_quality view (join the data_quality table with the source_table)
    view_name = f"{base_table}_data_quality"
    log.important(f"Creating Data Quality View {view_name}...")
    create_view_query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT A.*, B.data_quality
    FROM {source_table} A
    LEFT JOIN {data_quality_table} B
    ON A.{id_column} = B.{id_column}
    """

    # Execute the CREATE VIEW query
    data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import DataSource, FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a default data_quality view
    create_data_quality_view(fs.data_source, "id")
