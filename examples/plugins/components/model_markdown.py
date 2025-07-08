"""A Markdown Plugin Example for details/information about Models"""

import logging

# Dash Imports
from dash import html, dcc

# Workbench Imports
from workbench.api import Model
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the Workbench logger
log = logging.getLogger("workbench")


class MyModelMarkdown(PluginInterface):
    """MyModelMarkdown Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.MODEL
    plugin_input_type = PluginInputType.MODEL

    def create_component(self, component_id: str) -> html.Div:
        """Create a Model Markdown Component without any data

        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Graph: The EndpointTurbo Component
        """
        self.component_id = component_id
        self.container = html.Div(
            id=self.component_id,
            children=[
                html.H3(id=f"{self.component_id}-header", children="Model: Loading..."),
                dcc.Markdown(id=f"{self.component_id}-details"),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-details", "children"),
        ]

        # Return the container
        return self.container

    def update_properties(self, model: Model, **kwargs) -> list:
        """Update the properties for this plugin component

        Args:
            model (Model): An instantiated Model object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Model Markdown Plugin with Model: {model.name} and kwargs: {kwargs}")

        # Update the html header
        header = f"Model: {model.name}"

        # Make Markdown for the model summary
        summary = model.summary()
        markdown = ""
        for key, value in summary.items():

            # Chop off the "workbench_" prefix
            key = key.replace("workbench_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        # Return the updated property values for the plugin
        return [header, markdown]


# Unit Test for the Plugin
if __name__ == "__main__":
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(MyModelMarkdown).run()
