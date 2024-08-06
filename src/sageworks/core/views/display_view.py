"""Create a Display View: A View with a subset of columns for display purposes"""

import logging
from typing import Union

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


def create_display_view(
    data_source: DataSource,
    column_list: Union[list[str], None] = None,
    column_limit: int = 30,
    source_table: str = None,
):
    """Create a Display View: A View with a subset of columns for display purposes

    Args:
        data_source (DataSource): The DataSource object
        column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
        column_limit (int, optional): The max number of columns to include. Defaults to 30.
        source_table_name (str, optional): The table/view to create the view from. Defaults to data_source base table.
    """

    # Set the source_table to create the view from
    base_table = data_source.get_table_name()
    source_table = source_table if source_table else base_table

    # Create the display view table name
    view_name = f"{base_table}_display"

    log.important(f"Creating Display View {view_name}...")

    # If the user doesn't specify columns, then we'll limit the columns
    if column_list is None:
        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        column_list = [col for col in data_source.column_names() if col not in aws_cols]

        # Limit the number of columns
        column_list = column_list[:column_limit]

    # Enclose each column name in double quotes
    sql_columns = ", ".join([f'"{column}"' for column in column_list])

    # Create the view query
    create_view_query = f"""
       CREATE OR REPLACE VIEW {view_name} AS
       SELECT {sql_columns} FROM {source_table}
       """

    # Execute the CREATE VIEW query
    data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import DataSource, FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a default display view
    create_display_view(fs.data_source)

    # Create a display view with specific columns
    display_columns = fs.column_names()[:5]
    create_display_view(fs.data_source, display_columns)
