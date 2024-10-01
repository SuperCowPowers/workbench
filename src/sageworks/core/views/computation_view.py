"""ComputationView Class: Create a View with a subset of columns for display purposes"""

from typing import Union

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views.column_subset_view import ColumnSubsetView
from sageworks.core.views.view import View


class ComputationView(ColumnSubsetView):
    """ComputationView Class: Create a View with a subset of columns for computation purposes

    Common Usage:
        ```python
        # Create a default ComputationView
        fs = FeatureSet("test_features")
        comp_view = ComputationView.create(fs)
        df = comp_view.pull_dataframe()

        # Create a ComputationView with a specific set of columns
        comp_view = ComputationView.create(fs, column_list=["my_col1", "my_col2"])

        # Query the view
        df = comp_view.query(f"SELECT * FROM {comp_view.table} where prediction > 0.5")
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
        """Factory method to create and return a ComputationView instance.

        Args:
            artifact (Union[DataSource, FeatureSet]): The DataSource or FeatureSet object
            source_table (str, optional): The table/view to create the view from. Defaults to None
            column_list (Union[list[str], None], optional): A list of columns to include. Defaults to None.
            column_limit (int, optional): The max number of columns to include. Defaults to 30.

        Returns:
            Union[View, None]: The created View object (or None if failed to create the view)
        """
        # Use the create logic directly from ColumnSubsetView with the "computation" view name
        return ColumnSubsetView.create("computation", artifact, source_table, column_list, column_limit)


if __name__ == "__main__":
    """Exercise the Computation View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a ComputationView
    compute_view = ComputationView.create(fs)

    # Pull the computation data
    df = compute_view.pull_dataframe()
    print(df.head())
