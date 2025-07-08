"""A Markdown Component for details/information about Endpoints"""

import logging

# Dash Imports
from dash import html, dcc

# Workbench Imports
from workbench.api import Endpoint
from workbench.utils.markdown_utils import health_tag_markdown
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the Workbench logger
log = logging.getLogger("workbench")


class EndpointDetails(PluginInterface):
    """Endpoint Details Composite Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.ENDPOINT

    def __init__(self):
        """Initialize the EndpointDetails plugin class"""
        self.component_id = None
        self.current_endpoint = None

        # Call the parent class constructor
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.component_id = component_id
        container = html.Div(
            id=self.component_id,
            children=[
                html.H3(id=f"{self.component_id}-header", children="Endpoint: Loading..."),
                dcc.Markdown(id=f"{self.component_id}-details"),
            ],
        )

        # Fill in plugin properties
        self.properties = [(f"{self.component_id}-header", "children"), (f"{self.component_id}-details", "children")]

        # Return the container
        return container

    def update_properties(self, endpoint: Endpoint, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            endpoint (Endpoint): An instantiated Endpoint object
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        log.important(f"Updating Plugin with Endpoint: {endpoint.name} and kwargs: {kwargs}")

        # Update the header and the details
        self.current_endpoint = endpoint
        header = f"{self.current_endpoint.name}"
        details = self.endpoint_details()

        # Return the updated property values for the plugin
        return [header, details]

    def endpoint_details(self):
        """Construct the markdown string for the endpoint details

        Returns:
            str: A markdown string
        """
        # Get these fields from the endpoint
        show_fields = ["health_tags", "input", "status", "instance", "variant"]

        # Construct the markdown string
        summary = self.current_endpoint.details()
        markdown = ""
        for key in show_fields:

            # Special case for the health tags
            if key == "health_tags":
                markdown += health_tag_markdown(summary.get(key, []))
                continue

            # Get the value
            value = summary.get(key, "-")

            # If the value is a list, convert it to a comma-separated string
            if isinstance(value, list):
                value = ", ".join(value)

            # Chop off the "workbench_" prefix
            key = key.replace("workbench_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        return markdown


if __name__ == "__main__":
    # This class takes in endpoint details and generates a details Markdown component
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(EndpointDetails).run()
