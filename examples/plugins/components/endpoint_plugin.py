"""An example Endpoint plugin component"""

import logging
from dash import dcc


# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.api.endpoint import Endpoint


# Get the Workbench logger
log = logging.getLogger("workbench")


class MyEndpointPlugin(PluginInterface):
    """MyEndpointPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.ENDPOINT
    plugin_input_type = PluginInputType.ENDPOINT

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

    def update_properties(self, endpoint: Endpoint, **kwargs) -> list:
        """Create a Endpoint Plugin Figure

        Args:
            endpoint (Endpoint): An instantiated Endpoint object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Model Plugin with Model: {endpoint.uuid} and kwargs: {kwargs}")
        endpoint_name = f"Endpoint: {endpoint.uuid}"
        text_figure = self.display_text(endpoint_name, figure_height=100)

        # Return the updated property values
        return [text_figure]


if __name__ == "__main__":
    # A Unit Test for the Plugin
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(MyEndpointPlugin, test_type="endpoint").run()
