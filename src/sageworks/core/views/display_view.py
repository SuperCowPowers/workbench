"""DisplayView Class: Create a View with a subset of columns for display purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.column_subset_view import ColumnSubsetView


class DisplayView(ColumnSubsetView):
    """DisplayView Class: Create a View with a subset of columns for display purposes"""

    def __init__(self, artifact: Union[DataSource, FeatureSet], source_table: str = None):
        """Initialize the ColumnSubsetView

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
        """
        super().__init__("display", artifact, source_table)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a DisplayView
    view_maker = DisplayView(fs)
    display_view = view_maker.create()

    # Pull the display data
    df = display_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    columns = [col for col in fs.columns if col not in aws_cols]
    test_view = view_maker.create(column_list=columns)
    print(test_view.pull_dataframe().head())
