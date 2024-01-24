"""ModelPluginView is a tailored view of the Model Summary data + Details"""
import pandas as pd
import json

# SageWorks Imports
from sageworks.views.model_web_view import ModelWebView
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker, ServiceCategory


class ModelPluginView(ModelWebView):
    def __init__(self):
        """ModelPluginView pulls Model metadata and populates a Details Panel"""
        # Call SuperClass Initialization
        super().__init__()

        # We're using the AWS Service Broker to get additional information about the models
        self.aws_broker = AWSServiceBroker()

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

    def models_summary(self) -> pd.DataFrame:
        """Get summary data about the SageWorks Models"""

        # We get the summary dataframe from our parent class
        model_df = super().models_summary()

        # We pull a dictionary of model metrics from our internal method
        model_metrics = self._model_metrics()

        # Now 'join' the metrics to the model_df
        model_df["metrics"] = model_df["uuid"].map(model_metrics)
        return model_df

    def _model_metrics(self) -> dict[str]:
        """Internal: Get the metrics for all models and return a dictionary

        Returns:
            dict: In the form {model_uuid: metrics, model_uuid_2: metrics, ...}

        Notes:
            - There will not be keys for models without metrics
            - Metrics are in JSON.dumps format, so those will have to be unpacked
        """
        model_metrics = dict()

        # Now we use the AWS Service Broker to get additional information about
        # the Models. Specifically, we want to grab any inference metrics
        model_data = self.aws_broker.get_metadata(ServiceCategory.MODELS)
        for model_group_name, model_list in model_data.items():
            if not model_list:
                continue
            latest_model = model_list[0]
            sageworks_meta = latest_model.get("sageworks_meta", {})
            metrics = sageworks_meta.get("sageworks_inference_metrics")
            if metrics:
                model_metrics[latest_model["ModelPackageGroupName"]] = json.dumps(
                    pd.DataFrame.from_dict(metrics).to_dict(orient="records")
                )

        return model_metrics


if __name__ == "__main__":
    # Exercising the ModelPluginView
    import time

    # Create the class and get the AWS Model details
    model_view = ModelPluginView()

    # List the Models
    print("ModelsSummary:")
    summary = model_view.view_data()
    print(summary.head())

    # Give any broker threads time to finish
    time.sleep(1)
