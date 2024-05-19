"""A Markdown Component for details/information about Pipelines"""

import logging

# Dash Imports
from dash import html, dcc

# SageWorks Imports
from sageworks.api.pipeline import Pipeline
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class PipelineDetails(PluginInterface):
    """Pipeline Details Markdown Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.PIPELINE

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        container = html.Div(
            id=component_id,
            children=[
                html.H3(id=f"{component_id}-header", children="Pipeline: Loading..."),
                dcc.Markdown(id=f"{component_id}-details"),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{component_id}-header", "children"),
            (f"{component_id}-details", "children"),
        ]

        # Return the container
        return container

    def update_properties(self, pipeline: Pipeline, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            pipeline (Pipeline): An instantiated Pipeline object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Plugin with Pipeline: {pipeline.name} and kwargs: {kwargs}")

        # Update the header and the details
        header = f"{pipeline.name}"
        # pipeline_data = pipeline.get_pipeline_data()
        details = "**Details:**\n"
        details += f"**Name:** {pipeline.name}\n"

        # Return the updated property values for the plugin
        return [header, details]


if __name__ == "__main__":
    # This class takes in pipeline details and generates a details Markdown component
    from sageworks.web_components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(PipelineDetails).run()
