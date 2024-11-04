"""ColumnSubsetView Class: Create A View with a subset of columns"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.create_view import CreateView
from sageworks.core.views.view import View
from sageworks.core.views.view_utils import get_column_list


class ColumnSubsetView(CreateView):
    """ColumnSubsetView Class: Create a View with a subset of columns

    Common Usage:
        ```python
        # Create a ColumnSubsetView with a specific set of columns
        my_view = ColumnSubsetView.create("my_view", fs, column_list=["my_col1", "my_col2"])

        # Query the view
        df = my_view.query(f"SELECT * FROM {my_view.table} where residual > 0.8")
        ```
    """

    def __init__(self, view_name: str, artifact: Union[DataSource, FeatureSet], source_table: str = None):
        """Initialize the ColumnSubsetView

        Args:
            view_name (str): The name of the view
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
        """
        super().__init__(view_name, artifact, source_table)

    @classmethod
    def create(
        cls,
        view_name: str,
        artifact: Union[DataSource, FeatureSet],
        source_table: str = None,
        column_list: Union[list[str], None] = None,
        column_limit: int = 30,
    ) -> Union[View, None]:
        """Factory method to create and return a ColumnSubsetView instance.

        Args:
            view_name (str): The name of the view
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        # Instantiate the ColumnSubsetView
        instance = cls(view_name, artifact, source_table)

        # Get the columns from the source table
        source_table_columns = get_column_list(instance.data_source, instance.source_table)

        # If the user doesn't specify columns, then we'll grab columns from data_source with limit
        if column_list is None:
            column_list = source_table_columns[:column_limit]
        else:
            column_list = [col for col in column_list if col in source_table_columns]

        # Drop any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        column_list = [col for col in column_list if col not in aws_cols]

        # Enclose each column name in double quotes
        sql_columns = ", ".join([f'"{column}"' for column in column_list])

        # Sanity check the columns
        if not sql_columns:
            instance.log.critical(f"{artifact.uuid} No columns to create view...")
            return None

        # Create the view query
        create_view_query = f"""
            CREATE OR REPLACE VIEW "{instance.table}" AS
            SELECT {sql_columns} FROM "{instance.source_table}"
        """

        # Execute the CREATE VIEW query
        instance.data_source.execute_statement(create_view_query)

        # Return the View
        return View(instance.data_source, instance.view_name, auto_create_view=False)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a ColumnSubsetView
    test_view = ColumnSubsetView.create("test_subset", fs)

    # Pull the display data
    df = test_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    columns = ["id", "name", "age", "height", "weight"]
    test_view = ColumnSubsetView.create("test_subset", fs, column_list=columns)
    print(test_view.pull_dataframe(head=True))

    # Delete the View
    test_view.delete()
