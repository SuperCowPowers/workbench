"""ModelWebView pulls Model metadata from the AWS Service Broker with Details Panels on each Model"""

import pandas as pd

# SageWorks Imports
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
from sageworks.core.artifacts.model_core import ModelCore


class ModelWebView(ArtifactsWebView):
    def __init__(self):
        """ModelWebView pulls Model metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the Models Summary
        self.models_df = self.models_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        self.models_df = self.models_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Models View Data
        """
        return self.models_df

    def model_details(self, model_uuid: str) -> (dict, None):
        """Get all the details for the given Model UUID
        Args:
            model_uuid(str): The UUID of the Model
        Returns:
            dict: The details for the given Model (or None if not found)
        """
        model = ModelCore(model_uuid)
        if not model.exists():
            return {"Status": "Not Found"}
        elif not model.ready():
            return {"health_tags": model.get_health_tags()}

        # Return the Model Details
        return model.details()


if __name__ == "__main__":
    # Exercising the ModelWebView
    import time
    from pprint import pprint

    # Create the class and get the AWS Model details
    model_view = ModelWebView()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())

    # Get the details for the first Model
    my_model_uuid = summary["uuid"][0]
    print("\nModelDetails:")
    details = model_view.model_details(my_model_uuid)
    pprint(details)

    # Give any broker threads time to finish
    time.sleep(1)
