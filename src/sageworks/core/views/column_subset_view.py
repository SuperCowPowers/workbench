"""ColumnSubsetView Class: Create A View with a subset of columns"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.create_view import CreateView
from sageworks.core.views.view import View
from sageworks.core.views.view_utils import get_column_list


class ColumnSubsetView(CreateView):
    """ColumnSubsetView Class: Create a View with a subset of columns"""

    def __init__(self, artifact: Union[DataSource, FeatureSet], view_name: str = None):
        """Initialize the ColumnSubsetView

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            view_name (str): The name of the generated view
        """
        super().__init__(artifact, view_name)

    def create_view(
        self,
        column_list: Union[list[str], None] = None,
        column_limit: int = 30,
        source_table: str = None,
    ) -> Union[View, None]:
        """Create the View: A View with a subset of columns

        Args:
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.
            source_table (str, optional): The table/view to create the view from. Defaults to base table.

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """

        # Set the source_table to create the view from
        source_table = source_table if source_table else self.base_table

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
            self.log.critical(f"{self.data_source_name} No columns to create view...")
            return None

        # Create the view query
        create_view_query = f"""
           CREATE OR REPLACE VIEW {self.view_table_name} AS
           SELECT {sql_columns} FROM {source_table}
           """

        # Execute the CREATE VIEW query
        self.data_source.execute_statement(create_view_query)

        # Return the View
        return View(self.data_source, self.view_name, auto_create=False)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a ColumnSubsetView
    column_subset = ColumnSubsetView(fs, "test")
    test_view = column_subset.create_view()

    # Pull the display data
    df = test_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    columns = ["id", "name", "age", "height", "weight"]
    test_view = column_subset.create_view(column_list=columns)
    print(test_view.pull_dataframe(head=True))
