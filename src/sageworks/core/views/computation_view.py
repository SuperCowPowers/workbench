"""ComputationView Class: Create a View with a subset of columns for display purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.column_subset_view import ColumnSubsetView


class ComputationView(ColumnSubsetView):
    """ComputationView Class: Create a View with a subset of columns for display purposes"""

    def __init__(self, artifact: Union[DataSource, FeatureSet], source_table: str = None):
        """Initialize the ColumnSubsetView

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
        """
        super().__init__("computation", artifact, source_table)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a ComputationView
    view_maker = ComputationView(fs)
    compute_view = view_maker.create()

    # Pull the computation data
    df = compute_view.pull_dataframe()
    print(df.head())
