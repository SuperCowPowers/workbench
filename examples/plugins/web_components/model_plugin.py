"""An Example Model plugin component"""

from dash import dcc
import plotly.graph_objects as go
import random


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.model import Model


class MyModelPlugin(PluginInterface):
    """MyModelPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a EndpointTurbo Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_contents(self, model: Model) -> go.Figure:
        """Create a Figure for the plugin.
        Args:
            model (Model): An instantiated Model object
        Returns:
            go.Figure: A Plotly Figure object
        """
        model_name = f"Model: {model.uuid}"

        # Generate random values for the pie chart
        pie_values = [random.randint(10, 30) for _ in range(3)]

        # Create a pie chart with the endpoint name as the title
        fig = go.Figure(data=[go.Pie(labels=["A", "B", "C"], values=pie_values)], layout=go.Layout(title=model_name))
        return fig


if __name__ == "__main__":
    # This class takes in model details and generates a pie chart
    from sageworks.api.model import Model

    # Instantiate an Endpoint
    model = Model("abalone-regression")

    # Instantiate the EndpointTurbo class
    my_plugin = MyModelPlugin()

    # Generate the figure
    fig = my_plugin.update_contents(model)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
