"""Base View Class: This class is simply a 'pass through' to the data source table"""

# SageWorks Imports
from sageworks.core.views.view import View, ViewType


class BaseView(View):
    """Base View Class: This class is simply a 'pass through' to the data source table"""

    def __init__(self, data_source):
        """Initialize the Base View

        Args:
            data_source (DataSource): The DataSource object
        """
        super().__init__(data_source, ViewType.BASE)

    def create_view(self):
        """Create the Base View (which does nothing)"""
        self.log.important("Creating Base View...")


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a Base View
    base_view = BaseView(fs.data_source)
    print(base_view)

    # Pull the raw data
    df = base_view.pull_dataframe()
    print(df.head())
