"""A Model Plot component that switches display based on model type. The plot
will be a confusion matrix for a classifier and a regression plot for a regressor
"""

from dash import dcc
import plotly.graph_objects as go

# Workbench Imports
from workbench.api import Model
from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
from workbench.web_interface.components.regression_plot import RegressionPlot


class ModelPlot(ComponentInterface):
    """Model Metrics Components"""

    def create_component(self, component_id: str) -> dcc.Graph:
        # Initialize an empty plot figure
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_properties(self, model: Model, inference_run: str) -> go.Figure:
        """Create a Model Plot Figure based on the model type.
        Args:
            model (Model): Workbench Model object
            inference_run (str): Inference run capture UUID
        Returns:
            plotly.graph_objs.Figure: A Figure object containing the model metrics.
        """

        # Get model details
        model_details = model.details()

        # If the model details are empty then return a message
        if model_details is None:
            return self.display_text("Model Details are Empty")

        # Based on the model type, we'll generate a different plot
        model_type = model_details.get("model_type")
        if model_type == "classifier":
            return ConfusionMatrix().update_properties(model, inference_run=inference_run)[0]
        elif model_type in ["regressor", "quantile_regressor"]:
            return RegressionPlot().update_properties(model, inference_run)
        else:
            return self.display_text(f"Model Type: {model_type}\n\n Awesome Plot Coming Soon!")


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    from workbench.api.model import Model

    m = Model("wine-classification")
    inference_run = "model_training"

    # Instantiate the ModelPlot class
    model_plot = ModelPlot()

    # Generate the figure
    fig = model_plot.update_properties(m, inference_run)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
