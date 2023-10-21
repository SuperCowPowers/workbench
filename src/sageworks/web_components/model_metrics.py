"""A Model Metrics component that switches display based on model type"""
from dash import dcc
import plotly.graph_objects as go

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.web_components.confusion_matrix import ConfusionMatrix
from sageworks.web_components.regression_plot import RegressionPlot


class ModelMetrics(ComponentInterface):
    """Model Metrics Components"""

    def create_component(self, component_id: str) -> dcc.Graph:
        # Initialize an empty plot figure
        return dcc.Graph(id=component_id, figure=self.message_figure("Waiting for Data..."))

    def generate_component_figure(self, model_details: dict) -> go.Figure:
        """Create a Model Metrics Figure for the numeric columns in the dataframe
        Args:
            model_details (dict): The model details dictionary
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the model metrics.
        """

        # If the model details are empty then return a message
        if model_details is None:
            return self.message_figure("Model Details are Empty")

        # Based on the model type, we'll generate a different plot
        model_type = model_details.get("model_type")
        if model_type == "classifier":
            return ConfusionMatrix().generate_component_figure(model_details)
        elif model_type == "regressor":
            return RegressionPlot().generate_component_figure(model_details)
        else:
            return self.message_figure("Unknown Model Type")


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    from sageworks.artifacts.models.model import Model

    # m = Model("abalone-regression")
    m = Model("wine-classification")
    model_details = m.details()

    # Instantiate the ConfusionMatrix class
    reg_plot = ModelMetrics()

    # Generate the figure
    fig = reg_plot.generate_component_figure(model_details)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
