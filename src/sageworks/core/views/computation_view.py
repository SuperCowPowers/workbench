"""ComputationView Class: Create a View with a subset of columns for display purposes"""

# SageWorks Imports
from sageworks.core.views.column_subset_view import ColumnSubsetView


class ComputationView(ColumnSubsetView):
    """ComputationView Class: Create a View with a subset of columns for display purposes"""

    def __init__(self):
        """Initialize the ComputationView"""
        super().__init__()

    def get_view_name(self) -> str:
        """Get the name of the view"""
        return "computation"


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a ComputationView
    view_maker = ComputationView()
    compute_view = view_maker.create_view(fs)

    # Pull the computation data
    df = compute_view.pull_dataframe()
    print(df.head())
