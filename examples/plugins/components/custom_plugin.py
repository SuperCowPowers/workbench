"""A Custom plugin component"""

from dash import dcc


# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.api.model import Model


class CustomPlugin(PluginInterface):
    """CustomPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.CUSTOM
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a CustomPlugin Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.component_id = component_id
        self.container = dcc.Graph(id=component_id, figure=self.display_text("Waiting for Data..."))

        # Fill in plugin properties
        self.properties = [(self.component_id, "figure")]

        # Return the container
        return self.container

    def update_properties(self, model: Model, **kwargs) -> list:
        """Update the CustomPlugin property values

        Args:
            model (Model): An instantiated Endpoint object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        model_name = f"Model: {model.uuid}"
        text_figure = self.display_text(model_name, figure_height=100)
        return [text_figure]


if __name__ == "__main__":
    # A Unit Test for the Plugin
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(CustomPlugin).run()
