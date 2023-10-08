"""ModelWebView pulls Model metadata from the AWS Service Broker with Details Panels on each Model"""
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.models.model import Model


class ModelWebView(ArtifactsWebView):
    def __init__(self):
        """ModelWebView pulls Model metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # DataFrame of the Models Summary
        self.models_df = self.models_summary()

    def refresh(self):
        """Refresh the data from the AWS Service Broker"""
        super().refresh()
        self.models_df = self.models_summary()

    def view_data(self) -> pd.DataFrame:
        """Get all the data that's useful for this view

        Returns:
            pd.DataFrame: DataFrame of the Models View Data
        """
        return self.models_df

    def model_details(self, model_index: int) -> (dict, None):
        """Get all the details for the given Model Index"""
        uuid = self.model_name(model_index)
        model = Model(uuid)
        return model.details()

    def model_metrics(self, model_index: int) -> (dict, None):
        """Get the Model Training Metrics for the given Model Index"""
        uuid = self.model_name(model_index)
        model = Model(uuid)
        return model.model_metrics()

    def confusion_matrix(self, model_index: int) -> (dict, None):
        """Get the Model Training Metrics for the given Model Index"""
        uuid = self.model_name(model_index)
        model = Model(uuid)
        return model.confusion_matrix()

    def model_name(self, model_index: int) -> (str, None):
        """Helper method for getting the model name for the given Model Index"""
        if not self.models_df.empty and model_index < len(self.models_df):
            data_uuid = self.models_df.iloc[model_index]["uuid"]
            return data_uuid
        else:
            return None


if __name__ == "__main__":
    # Exercising the ModelWebView
    from pprint import pprint

    # Create the class and get the AWS Model details
    model_view = ModelWebView()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())

    # Get the details for the first Model
    print("\nModelDetails:")
    details = model_view.model_details(0)
    pprint(details)
