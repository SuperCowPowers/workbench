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

    def __init__(self):
        self.container = None

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a CustomPlugin Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.container = dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))
        return self.container

    def update_contents(self, endpoint: Endpoint, **kwargs):
        """Create a CustomPlugin Figure
        Args:
            endpoint (Endpoint): An instantiated Endpoint object
            **kwargs: Additional keyword arguments (unused)
        """
        endpoint_name = f"Endpoint: {endpoint.uuid}"
        self.container.figure = self.display_text(endpoint_name)


if __name__ == "__main__":
    # This class takes in a model object
    from dash import html, Dash

    # Test if the Plugin Class is a valid PluginInterface
    assert issubclass(MyEndpointPlugin, PluginInterface)

    # Instantiate the CustomPlugin class
    my_plugin = MyEndpointPlugin()
    my_component = my_plugin.create_component("endpoint_plugin")

    # Give the Endpoint object to the plugin
    endpoint = Endpoint("abalone-regression-end")
    my_plugin.update_contents(endpoint)

    # Initialize Dash app
    app = Dash(__name__)

    app.layout = html.Div([my_component])
    app.run(debug=True)
