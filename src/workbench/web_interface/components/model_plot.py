"""A Model Plot component that switches display based on model type. The plot
will be a confusion matrix for a classifier and a regression plot for a regressor
"""

from dash import dcc
import plotly.graph_objects as go

# Workbench Imports
from workbench.api import Model, ModelType
from workbench.web_interface.components.component_interface import ComponentInterface
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot


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
            go.Figure: A Plotly Figure (either a Confusion Matrix or Scatter Plot)
        """

        # Based on the model type, we'll generate a different plot
        if model.model_type == ModelType.CLASSIFIER:
            return ConfusionMatrix().update_properties(model, inference_run=inference_run)[0]
        elif model.model_type in [ModelType.REGRESSOR, ModelType.QUANTILE_REGRESSOR]:
            df = model.get_inference_predictions(inference_run)
            if df is None:
                return self.display_text("No Data")

            # Calculate the distance from the diagonal for each point
            target = model.target()
            df["prediction_error"] = abs(df["prediction"] - df[target])
            return ScatterPlot().update_properties(df, color="prediction_error", regression_line=True)[0]
        else:
            return self.display_text(f"Model Type: {model.model_type}\n\n Awesome Plot Coming Soon!")


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    from workbench.api.model import Model

    m = Model("abalone-regression")
    inference_run = "model_training"

    # Instantiate the ModelPlot class
    model_plot = ModelPlot()

    # Generate the figure
    fig = model_plot.update_properties(m, inference_run)

    # Show the figure
    fig.show()
