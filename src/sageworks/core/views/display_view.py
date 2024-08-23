"""DisplayView Class: A View with a subset of columns for display purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource
from sageworks.core.views.view import View, ViewType
from sageworks.core.views.view_utils import get_column_list


class DisplayView(View):
    """DisplayView Class: A View with a subset of columns for display purposes"""

    def __init__(self, data_source: DataSource):
        """Initialize the DisplayView

        Args:
            data_source (DataSource): The DataSource object
        """
        super().__init__(data_source, ViewType.DISPLAY)

    def create_view(
        self,
        column_list: Union[list[str], None] = None,
        column_limit: int = 30,
        source_table: str = None,
    ):
        """Create a Display View: A View with a subset of columns for display purposes

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
            source_table (str, optional): The table/view as a source for view. Defaults to data_source base table.
        """
        self.log.important("Creating Display View...")

        # Set the source_table to create the view from
        source_table = source_table if source_table else self.base_table

        # Create the display view table name
        view_name = f"{self.base_table}_display"

        self.log.important(f"Creating Display View {view_name}...")

        # If the user doesn't specify columns, then we'll limit the columns
        if column_list is None:
            # Drop any columns generated from AWS
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
            source_table_columns = get_column_list(self.data_source, source_table)
            column_list = [col for col in source_table_columns if col not in aws_cols]

            # Limit the number of columns
            column_list = column_list[:column_limit]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Sanity check the columns
        if not sql_columns:
            self.log.critical(f"{self.data_source_name} No columns to create display view...")
            return

        # Create the view query
        create_view_query = f"""
           CREATE OR REPLACE VIEW {view_name} AS
           SELECT {sql_columns} FROM {source_table}
           """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a DisplayView
    display_view = DisplayView(fs)
    print(display_view)

    # Pull the display data
    df = display_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    columns = ["id", "name", "age", "height", "weight"]
    display_view.create_view(column_list=columns)
    print(display_view.pull_dataframe(head=True))
