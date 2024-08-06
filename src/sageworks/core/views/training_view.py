"""Create Training Views: All columns + a training column with 0/1 values for training/validation"""

import logging
from typing import Union

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


def create_training_view(
    data_source: DataSource, id_column: str, holdout_ids: Union[list[str], None] = None, source_table: str = None
):
    """Create a training view that marks hold out ids

    Args:
        data_source (DataSource): The DataSource object
        id_column (str): The name of the id column
        holdout_ids (Union[list[str], None], optional): A list of holdout ids. Defaults to None.
        source_table_name (str, optional): The table/view to create the view from. Defaults to data_source base table.
    """

    # Set the source_table to create the view from
    base_table = data_source.get_table_name()
    source_table = source_table if source_table else base_table

    # Create the training view table name
    view_name = f"{base_table}_training"

    # If we don't have holdout ids, create a default training view
    if not holdout_ids:
        _default_training_view(data_source, id_column, view_name)
        return

    log.important(f"Creating Training View {view_name}...")

    # Format the list of hold out ids for SQL IN clause
    if holdout_ids and all(isinstance(id, str) for id in holdout_ids):
        formatted_holdout_ids = ", ".join(f"'{id}'" for id in holdout_ids)
    else:
        formatted_holdout_ids = ", ".join(map(str, holdout_ids))

    # Drop any columns generated from AWS
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    column_list = [col for col in data_source.column_names() if col not in aws_cols]

    # Enclose each column name in double quotes
    sql_columns = ", ".join([f'"{column}"' for column in column_list])

    # Construct the CREATE VIEW query
    create_view_query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT {sql_columns}, CASE
        WHEN {id_column} IN ({formatted_holdout_ids}) THEN 0
        ELSE 1
    END AS training
    FROM {base_table}
    """

    # Execute the CREATE VIEW query
    data_source.execute_statement(create_view_query)


# This is an internal method that's used to create a default training view
def _default_training_view(data_source: DataSource, id_column: str, view_name: str):
    """Create a default view in Athena that assigns roughly 80% of the data to training

    Args:
        data_source (DataSource): The DataSource object
        id_column (str): The name of the id column
        view_name (str): The name of the view to create
    """
    log.important(f"Creating default Training View {view_name}...")

    # Drop any columns generated from AWS
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    column_list = [col for col in data_source.column_names() if col not in aws_cols]

    # Enclose each column name in double quotes
    sql_columns = ", ".join([f'"{column}"' for column in column_list])

    #    Construct the CREATE VIEW query with a simple modulo operation for the 80/20 split
    #    using the id column as the stable identifier for row numbering
    base_table = data_source.get_table_name()
    create_view_query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT {sql_columns}, CASE
        WHEN MOD(ROW_NUMBER() OVER (ORDER BY {id_column}), 10) < 8 THEN 1  -- Assign 80% to training
        ELSE 0  -- Assign roughly 20% to validation/test
    END AS training
    FROM {base_table}
    """

    # Execute the CREATE VIEW query
    data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import DataSource, FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")
    id_column = "id"

    # Create a default training view
    create_training_view(fs.data_source, id_column)

    # Create a training view with holdout ids (first 10)
    holdout_ids = list(range(10))
    create_training_view(fs.data_source, id_column, holdout_ids)
