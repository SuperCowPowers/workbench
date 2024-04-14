"""An Example Model plugin component"""

import logging
from dash import dcc
import random
import plotly.graph_objects as go

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.api.model import Model

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class ModelPlugin(PluginInterface):
    """MyModelPlugin Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> dcc.Graph:
        """Create a Model Component without any data.
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
        """Update the properties for the plugin.

        Args:
            model (Model): An instantiated Model object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Model Plugin with Model: {model.uuid} and kwargs: {kwargs}")
        model_name = f"Model: {model.uuid}"

        # Generate random values for the pie chart
        pie_values = [random.randint(10, 30) for _ in range(4)]

        # Create a pie chart with the endpoint name as the title
        pie_figure = go.Figure(
            data=[go.Pie(labels=["A", "B", "C", "D"], values=pie_values)], layout=go.Layout(title=model_name)
        )

        # Return the updated property values for the plugin
        return [pie_figure]


if __name__ == "__main__":
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(ModelPlugin).run()
