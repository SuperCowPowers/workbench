"""A Custom plugin component"""

from dash import dcc


# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.model import Model


class CustomPlugin(PluginInterface):
    """CustomPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    plugin_page = PluginPage.CUSTOM
    plugin_input_type = PluginInputType.MODEL

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

    def update_contents(self, model: Model, **kwargs):
        """Create a CustomPlugin Figure
        Args:
            model (Model): An instantiated Endpoint object
            **kwargs: Additional keyword arguments (unused)
        """
        model_name = f"Model: {model.uuid}"
        self.container.figure = self.display_text(model_name)


if __name__ == "__main__":
    # Test the custom plugin component
    from dash import html, Dash

    # Test if the Plugin Class is a valid PluginInterface
    assert issubclass(CustomPlugin, PluginInterface)

    # Instantiate the CustomPlugin class
    my_plugin = CustomPlugin()
    my_component = my_plugin.create_component("custom_plugin")

    # Give the model object to the plugin
    model = Model("abalone-regression")
    my_plugin.update_contents(model)

    # Initialize Dash app
    app = Dash(__name__)

    app.layout = html.Div([my_component])
    app.run(debug=True)
