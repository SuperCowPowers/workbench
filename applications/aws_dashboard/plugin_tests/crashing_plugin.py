"""A Plugin Component that Crashes for Testing"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import (
    PluginInterface,
    PluginType,
    PluginInputType,
    plugin_error_decorator,
)


class CrashingPlugin(PluginInterface):
    """CrashingPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_type = PluginType.MODEL
    plugin_input_type = PluginInputType.MODEL_DETAILS

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a CrashingPlugin Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The CrashingPlugin Component
        """
        return dcc.Graph(id=component_id, figure=self.message_figure("Waiting for Data..."))

    @plugin_error_decorator
    def generate_component_figure(self, model_details: dict) -> go.Figure:
        """Create a CrashingPlugin Figure for the numeric columns in the dataframe.
        Args:
            model_details (dict): The model details dictionary (see Model.details())
        Returns:
            go.Figure: A Figure object containing the confusion matrix.
        """

        # This is where the plugin crashes
        my_bad = model_details["bad_key"]

        # Create the nested pie chart plot with custom settings
        fig = go.Figure(my_bad)
        fig.update_layout(margin={"t": 10, "b": 10, "r": 10, "l": 10, "pad": 10}, height=400)

        return fig


if __name__ == "__main__":
    # This class takes in model details and generates a CrashingPlugin

    # Instantiate the CrashingPlugin class
    bad_plugin = CrashingPlugin()

    # Generate the figure
    my_model_details = {"key": "value"}
    fig = bad_plugin.generate_component_figure(my_model_details)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
