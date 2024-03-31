"""An example Endpoint plugin component"""

from dash import dcc
import plotly.graph_objects as go


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.endpoint import Endpoint


class MyEndpointPlugin(PluginInterface):
    """MyEndpointPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.ENDPOINT
    plugin_input_type = PluginInputType.ENDPOINT

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a CustomPlugin Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        return dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

    def update_contents(self, endpoint: Endpoint) -> go.Figure:
        """Create a CustomPlugin Figure
        Args:
            endpoint (Endpoint): An instantiated Endpoint object
        Returns:
            go.Figure: A Plotly Figure object
        """
        endpoint_name = f"Endpoint: {endpoint.uuid}"
        return self.display_text(endpoint_name, figure_height=200)


if __name__ == "__main__":
    # This class takes in a model object

    # Instantiate an Endpoint
    my_endpoint = Endpoint("abalone-regression-end")

    # Instantiate the EndpointTurbo class
    plugin = MyEndpointPlugin()

    # Generate the figure
    fig = plugin.update_contents(my_endpoint)

    # Apply dark theme
    fig.update_layout(template="plotly_dark")

    # Show the figure
    fig.show()
