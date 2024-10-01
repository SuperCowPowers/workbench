"""DisplayView Class: Create a View with a subset of columns for display purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.view import View
from sageworks.core.views.column_subset_view import ColumnSubsetView


class DisplayView(ColumnSubsetView):
    """DisplayView Class: Create a View with a subset of columns for display purposes

    Common Usage:
        ```python
        # Create a default DisplayView
        fs = FeatureSet("test_features")
        display_view = DisplayView.create(fs)
        df = display_view.pull_dataframe()

        # Create a DisplayView with a specific set of columns
        display_view = DisplayView.create(fs, column_list=["my_col1", "my_col2"])

        # Query the view
        df = display_view.query(f"SELECT * FROM {display_view.table} where awesome = 'yes'")
        ```
    """

    @classmethod
    def create(
        cls,
        artifact: Union[DataSource, FeatureSet],
        source_table: str = None,
        column_list: Union[list[str], None] = None,
        column_limit: int = 30,
    ) -> Union[View, None]:
        """Factory method to create and return a DisplayView instance.

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        # Use the create logic directly from ColumnSubsetView with the "display" view name
        return ColumnSubsetView.create("display", artifact, source_table, column_list, column_limit)


if __name__ == "__main__":
    """Exercise the Display View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a DisplayView
    display_view = DisplayView.create(fs)

    # Pull the display data
    df = display_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    columns = [col for col in fs.columns if col not in aws_cols]
    display_view = DisplayView.create(fs, column_list=columns)
    print(display_view.pull_dataframe().head())
