"""ModelPluginView is a tailored view of the Model Summary data + Details"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_model import CachedModel


class ModelPluginView(PageView):
    def __init__(self):
        """ModelPluginView pulls Model metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # We're using the Workbench Meta class to get information about models
        self.meta = CachedModel()

        # Call Refresh
        self.models_df = None
        self.refresh()  # Sets the self.models_df

    def refresh(self):
        """Refresh our data from the Workbench Meta Class"""

        # Note: This page is served on an AWS Web server and stays up 24/7.
        #       We want to make sure new models show up when they are created.
        self.models_df = self.models_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Models View Data
        """
        return self.models_df

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the Workbench Models"""
        models = self.meta.models()
        models["uuid"] = models["Model Group"]
        return models


if __name__ == "__main__":
    # Exercising the ModelPluginView

    # Create the class and get the AWS Model details
    model_view = ModelPluginView()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())
