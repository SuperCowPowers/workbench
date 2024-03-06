"""A Regression Plot component"""

from dash import dcc
import plotly.graph_objects as go
import plotly.express as px

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.api import Model


# This class is basically a specialized version of a Plotly Scatter Plot
# For heatmaps see (https://plotly.com/python/line-and-scatter/)
class RegressionPlot(ComponentInterface):
    """Regression Plot Component"""

    def create_component(self, component_id: str) -> dcc.Graph:
        # Initialize an empty scatter plot figure
        return dcc.Graph(id=component_id, figure=self.message_figure("Waiting for Data..."))

    def generate_component_figure(self, model: Model, inference_run: str = None) -> go.Figure:
        # Get predictions for specific inference
        df = model.inference_predictions(inference_run)

        if df is None:
            return self.message_figure("No Data")

        # Get the name of the actual field value column
        actual_col = [col for col in df.columns if col != "prediction"][0]

        # Calculate the distance from the diagonal for each point
        df["prediction_error"] = abs(df["prediction"] - df[actual_col])

        color_scale = [
            [0, "rgb(64,64,160)"],
            [0.35, "rgb(48, 140, 140)"],
            [0.65, "rgb(140, 140, 48)"],
            [1.0, "rgb(160, 64, 64)"],
        ]

        # Create the scatter plot with bigger dots
        fig = px.scatter(
            df,
            x=actual_col,
            y="prediction",
            size="prediction_error",
            size_max=20,
            color="prediction_error",
            color_continuous_scale=color_scale,
        )

        # Just fine tuning the dots on the scatter plot
        fig.update_traces(
            marker=dict(size=14, line=dict(width=1, color="Black")),
            selector=dict(mode="markers"),
        )

        # Add a diagonal line for reference
        min_val = min(df[actual_col].min(), df["prediction"].min())
        max_val = max(df[actual_col].max(), df["prediction"].max())
        fig.add_shape(
            type="line",
            line=dict(width=5, color="rgba(1.0, 1.0, 1.0, 0.5)"),
            x0=min_val,
            x1=max_val,
            y0=min_val,
            y1=max_val,
        )

        # Just some fine tuning of the plot
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=400)

        return fig


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    from sageworks.api import Model

    m = Model("abalone-regression")
    inference_run = "training_holdout"

    # Instantiate the ConfusionMatrix class
    reg_plot = RegressionPlot()

    # Generate the figure
    fig = reg_plot.generate_component_figure(m, inference_run)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
