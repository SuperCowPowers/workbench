"""Create Display Views: A View with a subset of columns for display purposes"""

import logging
from typing import Union

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


def create_display_view(data_source: DataSource, column_list: Union[list[str], None] = None, column_limit: int = 30):
    """Create a display view that shows a subset of columns

    Args:
        data_source (DataSource): The DataSource object
        column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
        column_limit (int, optional): The max number of columns to include. Defaults to 30.
    """

    # Create the training view table name
    base_table = data_source.get_table_name()
    view_name = f"{base_table}_display"

    log.important(f"Creating Display View {view_name}...")

    # If the user doesn't specify columns, then we'll limit the columns
    if column_list is None:
        column_list = data_source.column_names()[:column_limit]

    # Enclose each column name in double quotes
    sql_columns = ", ".join([f'"{column}"' for column in column_list])

    # Create the view query
    create_view_query = f"""
       CREATE OR REPLACE VIEW {view_name} AS
       SELECT {sql_columns} FROM {base_table}
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
