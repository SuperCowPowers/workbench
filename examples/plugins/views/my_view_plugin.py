"""MyViewPlugin is a tailored view of the Model Summary data + Details"""

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.api import Meta


class MyViewPlugin(PageView):
    def __init__(self):
        """MyViewPlugin pulls Model metadata"""

        # Call SuperClass Initialization
        super().__init__()

        # We're using the Workbench Meta class to get information about models
        self.meta = Meta()

    def refresh(self):
        """Refresh our data (for this example, we don't need to)"""
        pass

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Models Data
        """
        models = self.meta.models()
        models["uuid"] = models["Model Group"]  # uuid is needed for identifying the model
        return models


if __name__ == "__main__":
    # Exercising the MyViewPlugin
    import pandas as pd

    pd.options.display.max_columns = None
    pd.options.display.width = 1000

    # Create the class and get the AWS Model details
    model_view = MyViewPlugin()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())
