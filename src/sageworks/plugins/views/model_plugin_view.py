"""ModelPluginView is a tailored view of the Model Summary data + Details"""

import json
import pandas as pd

# SageWorks Imports
from sageworks.views.view import View
from sageworks.api.meta import Meta
from sageworks.api.model import Model


class ModelPluginView(View):
    def __init__(self):
        """ModelPluginView pulls Model metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # We're using the SageWorks Meta class to get information about models
        self.meta = Meta()

        # Call Refresh
        self.models_df = None
        self.refresh()  # Sets the self.models_df

    def refresh(self):
        """Refresh our data from the SageWorks Meta Class"""

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
        """Get summary data about the SageWorks Models"""

        # Note: The data we get from Meta is structured as follows:
        # {
        #    model_group_name: [model_1, model_2, ...],
        #    model_group_name_2: [model_1, model_2, ...], ...
        # }
        model_summary = []
        for model_group_name, model_list in self.meta.models().items():
            # Get Summary information for the 'latest' model in the model_list
            latest_model = model_list[0]
            sageworks_meta = latest_model.get("sageworks_meta", {})
            summary = {
                "uuid": latest_model["ModelPackageGroupName"],  # Required for selection
                "Name": latest_model["ModelPackageGroupName"],
                "Owner": sageworks_meta.get("sageworks_owner", "-"),
                "Model Type": sageworks_meta.get("sageworks_model_type"),
                "Created": latest_model.get("CreationTime"),
                "Ver": latest_model["ModelPackageVersion"],
                "Tags": sageworks_meta.get("sageworks_tags", "-"),
                "Input": sageworks_meta.get("sageworks_input", "-"),
                "Status": latest_model["ModelPackageStatus"],
                "Description": latest_model.get("ModelPackageDescription", "-"),
                "Metrics": self.get_performance_metrics(latest_model),
            }
            model_summary.append(summary)

        # Now return a dataframe of the model_summary
        return pd.DataFrame(model_summary)

    @staticmethod
    def get_performance_metrics(latest_model) -> list[dict]:
        """Get the metrics for a single model

        Args:
            latest_model (dict): The latest model from the SageWorks Meta Class

        Returns:
            list(dict): A list of metrics for the model
        """
        sageworks_meta = latest_model.get("sageworks_meta", {})
        metrics = sageworks_meta.get("sageworks_inference_metrics")
        return json.dumps(metrics) if metrics else []

    @staticmethod
    def model_details(model_uuid) -> dict:
        """Get the details for the given model

        Args:
            model_uuid (str): The uuid of the model to get details for

        Returns:
            dict: A dictionary of the model details
        """
        return Model(model_uuid).details()


if __name__ == "__main__":
    # Exercising the ModelPluginView

    # Create the class and get the AWS Model details
    model_view = ModelPluginView()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())
