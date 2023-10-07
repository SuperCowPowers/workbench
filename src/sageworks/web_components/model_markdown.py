"""A Markdown Component for details/information about Models"""
from dash import dcc

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface


class ModelMarkdown(ComponentInterface):
    """Model Markdown Component"""

    def create_component(self, component_id: str) -> dcc.Markdown:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            dcc.Markdown: The Dash Markdown Component
        """
        waiting_markdown = "*Waiting for data...*"
        return dcc.Markdown(id=component_id, children=waiting_markdown, dangerously_allow_html=True)

    def generate_markdown(self, model_details: dict) -> str:
        """Create the Markdown for the details/information about the DataSource or the FeatureSet
        Args:
            model_details (dict): A dictionary of information about the artifact
        Returns:
            str: A Markdown string
        """

        # Create simple markdown by iterating through the model_details dictionary
        # and adding a bullet point for each key/value pair
        model_name = model_details["uuid"]

        # Excluded keys from the model_details dictionary (and any keys that end with '_arn')
        exclude = ["size", "uuid", "confusion_matrix"]
        top_level_details = {
            key: value for key, value in model_details.items() if key not in exclude and not key.endswith("_arn")
        }

        # Now Loop through the model_details dictionary and create the markdown
        markdown = f"#### {model_name}\n"
        for key, value in top_level_details.items():
            markdown += f"**{key}:** {value}<br>"

        # Now let's add data associated with the Scoring and Metrics
        markdown += "\n#### Model Scores and Metrics\n"
        if "confusion_matrix" in model_details:
            markdown += f"**Confusion Matrix:** {model_details['confusion_matrix']}<br>"
        return markdown
