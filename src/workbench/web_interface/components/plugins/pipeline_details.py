"""A Markdown Component for details/information about Pipelines"""

import logging

# Dash Imports
from dash import html, dcc

# Workbench Imports
from workbench.api.pipeline import Pipeline
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType

# Get the Workbench logger
log = logging.getLogger("workbench")


class PipelineDetails(PluginInterface):
    """Pipeline Details Markdown Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.PIPELINE

    def __init__(self):
        """Initialize the PipelineDetails plugin class"""
        self.component_id = None
        self.current_pipeline = None

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
                html.H3(id=f"{self.component_id}-header", children="Pipeline: Loading..."),
                dcc.Markdown(id=f"{self.component_id}-details"),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{self.component_id}-header", "children"),
            (f"{self.component_id}-details", "children"),
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
        log.important(f"Updating Plugin with Pipeline: {pipeline.uuid} and kwargs: {kwargs}")

        # Update the header and the details
        self.current_pipeline = pipeline
        header = f"{self.current_pipeline.uuid}"
        details = self.pipeline_details()

        # Return the updated property values for the plugin
        return [header, details]

    def pipeline_details(self):
        """Construct the markdown string for the pipeline details

        Returns:
            str: A markdown string
        """

        # Grab the pipeline details and construct the markdown string
        details = self.current_pipeline.details()
        markdown = self.pipeline_to_markdown(details)
        return markdown

    def pipeline_to_markdown(self, pipeline_details: dict) -> str:
        """Convert pipeline details to a markdown string with hyperlinks and details.

        Args:
            pipeline_details (dict): A dictionary of pipeline details.

        Returns:
            str: A markdown string as a bulleted list.
        """
        markdown = ""

        # Each pipeline will have Workbench Artifact keys (data_source, feature_set, model, etc.)
        for key, value in pipeline_details.items():
            uuid = value.get("name")
            markdown += f"- {self._hyperlink(key, uuid)}\n"

        return markdown

    def _hyperlink(self, artifact_type: str, uuid: str) -> str:
        """Create a hyperlink for a Workbench artifact type and name.

        Args:
            artifact_type (str): The type of Workbench artifact (e.g., "data_source").
            uuid (str): The unique identifier for the artifact.

        Returns:
            str: A markdown hyperlink string.
        """
        # Convert underscores to CamelCase for display purposes
        artifact_type_display = artifact_type.title().replace("_", "")

        # Return the markdown hyperlink string (relative to the root)
        return f"[{artifact_type_display}({uuid})]({artifact_type}s?uuid={uuid})"

    def _dict_to_markdown(self, dictionary: dict, indent: int = 0) -> str:
        """Convert a dictionary to a markdown string with nested list formatting.

        Args:
            dictionary (dict): A dictionary to convert to markdown.
            indent (int): The current level of indentation (for nested lists).

        Returns:
            str: A markdown string.
        """
        markdown = ""
        prefix = "  " * indent + "- "  # Use "- " for Markdown nested list items

        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Add the key as a parent item and recurse for nested dictionary
                markdown += f"{prefix}**{key}:**\n"
                markdown += self.dict_to_markdown(value, indent + 1)
            elif isinstance(value, list):
                # Add the key as a parent item, then each list item
                markdown += f"{prefix}**{key}:**\n"
                for item in value:
                    markdown += f"{'  ' * (indent + 1)}- {item}\n"
            else:
                # Add a plain key-value pair
                markdown += f"{prefix}**{key}:** {value}\n"

        return markdown


if __name__ == "__main__":
    # This class takes in pipeline details and generates a details Markdown component
    from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

    # Run the Unit Test on the Plugin
    PluginUnitTest(PipelineDetails).run()
