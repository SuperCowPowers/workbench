"""DisplayView Class: Create a View with a subset of columns for display purposes"""

# SageWorks Imports
from sageworks.core.views.column_subset_view import ColumnSubsetView


class DisplayView(ColumnSubsetView):
    """DisplayView Class: Create a View with a subset of columns for display purposes"""

    def __init__(self):
        """Initialize the DisplayView"""
        super().__init__()

    def get_view_name(self) -> str:
        """Get the name of the view"""
        return "display"


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a DisplayView
    view_maker = DisplayView()
    display_view = view_maker.create_view(fs)

    # Pull the display data
    df = display_view.pull_dataframe()
    print(df.head())

    # Create a Display View with a subset of columns
    aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
    columns = [col for col in fs.column_names() if col not in aws_cols]
    test_view = view_maker.create_view(fs, column_list=columns)
    print(test_view.pull_dataframe().head())
