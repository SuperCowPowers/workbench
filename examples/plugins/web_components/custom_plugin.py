"""A Custom plugin component"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.model import Model


class CustomPlugin(PluginInterface):
    """CustomPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.CUSTOM
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a CustomPlugin Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_contents(self, model: Model) -> go.Figure:
        """Create a CustomPlugin Figure
        Args:
            model (Model): An instantiated Endpoint object
        Returns:
            go.Figure: A Plotly Figure object
        """
        model_name = f"Model: {model.uuid}"
        return self.display_text(model_name)


if __name__ == "__main__":
    # This class takes in a model object

    # Instantiate an Endpoint
    my_model = Model("abalone-regression")

    # Instantiate the EndpointTurbo class
    plugin = CustomPlugin()

    # Generate the figure
    fig = plugin.update_contents(my_model)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
