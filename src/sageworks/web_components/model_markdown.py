"""A Markdown Component for details/information about Models"""
import pandas as pd
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
        return dcc.Markdown(id=component_id, children=waiting_markdown, dangerously_allow_html=False)

    def generate_markdown(self, model_details: dict) -> str:
        """Create the Markdown for the details/information about the DataSource or the FeatureSet
        Args:
            model_details (dict): A dictionary of information about the artifact
        Returns:
            str: A Markdown string
        """

        # Create simple markdown by iterating through the model_details dictionary

        # Excluded keys from the model_details dictionary (and any keys that end with '_arn')
        exclude = ["size", "uuid"]
        top_level_details = {
            key: value for key, value in model_details.items() if key not in exclude and not key.endswith("_arn")
        }


        # Exclude dataframe values
        top_level_details = {
            key: value for key, value in top_level_details.items() if not isinstance(value, pd.DataFrame)
        }

        markdown = ""
        for key, value in top_level_details.items():
            # Escape square brackets
            if isinstance(value, (list, tuple)):
                value_str = str(value).replace("[", r"\[").replace("]", r"\]")
            else:
                value_str = str(value)

            # Add to markdown string
            markdown += f"**{key}:** {value_str}  \n"

        # Model Metrics
        markdown += "### Model Metrics  \n"
        if model_details["uuid"] == "abalone-regression":
            markdown += "**Test Data:** Abalone_Regression_Test_2023_10_11  \n"
            markdown += "**Test Data Hash:** ebea16fbc63574fe91dcac35a0b2432f  \n"
        else:
            markdown += "**Test Data:** Wine_Classification_Test_2023_09_03  \n"
            markdown += "**Test Data Hash:** cac35a0b2432febea16fbc63574fe91d  \n"

        # Grab the Metrics from the model details
        metrics = model_details.get("model_metrics")
        if metrics is None:
            markdown += ("  \nNo Data  \n")
        else:
            markdown += "  \n"
            markdown += metrics.to_markdown(index=False)

        return markdown


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    import dash
    from dash import dcc
    from dash import html
    from sageworks.artifacts.models.model import Model

    # Create the class and get the AWS FeatureSet details
    m = Model("abalone-regression")
    model_details = m.details()

    # Instantiate the DataDetailsMarkdown class
    ddm = ModelMarkdown()
    component = ddm.create_component("model_markdown")

    # Generate the markdown
    markdown = ddm.generate_markdown(model_details)

    # Initialize Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        component
    ])
    component.children = markdown

    if __name__ == '__main__':
        app.run_server(debug=True)