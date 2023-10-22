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

        # If the model details are empty then return a message
        if model_details is None:
            return "*No Data*"

        # Create simple markdown by iterating through the model_details dictionary

        # Excluded keys from the model_details dictionary (and any keys that end with '_arn')
        exclude = ["size", "uuid", "inference_meta", "model_info"]
        top_level_details = {
            key: value for key, value in model_details.items() if key not in exclude and not key.endswith("_arn")
        }

        # FIXME: Remove this later: Add the model info to the top level details
        model_info = model_details.get("model_info", {})
        prefixed_model_info = {f"model_{k}": v for k, v in model_info.items()}
        top_level_details.update(prefixed_model_info)

        # Exclude dataframe values
        top_level_details = {
            key: value for key, value in top_level_details.items() if not isinstance(value, pd.DataFrame)
        }

        # Construct the markdown string
        markdown = ""
        for key, value in top_level_details.items():
            # Escape square brackets
            if isinstance(value, (list, tuple)):
                value_str = str(value).replace("[", r"\[").replace("]", r"\]")
            else:
                value_str = str(value)

            # Add to markdown string
            markdown += f"**{key}:** {value_str}  \n"

        # Model Test Metrics
        markdown += "### Model Test Metrics  \n"
        meta_df = model_details.get("inference_meta")
        if meta_df is None:
            test_data = "AWS Training Capture"
            test_data_hash = " N/A "
            test_rows = " - "
            description = " - "
        else:
            inference_meta = meta_df.to_dict(orient="records")[0]
            test_data = inference_meta.get("test_data", " - ")
            test_data_hash = inference_meta.get("test_data_hash", " - ")
            test_rows = inference_meta.get("test_rows", " - ")
            description = inference_meta.get("description", " - ")

        # Add the markdown for the model test metrics
        markdown += f"**Test Data:** {test_data}  \n"
        markdown += f"**Data Hash:** {test_data_hash}  \n"
        markdown += f"**Test Rows:** {test_rows}  \n"
        markdown += f"**Description:** {description}  \n"

        # Grab the Metrics from the model details
        metrics = model_details.get("model_metrics")
        if metrics is None:
            markdown += "  \nNo Data  \n"
        else:
            markdown += "  \n"
            metrics = metrics.round(3)
            markdown += metrics.to_markdown(index=False)

        return markdown


if __name__ == "__main__":
    # This class takes in model details and generates a Confusion Matrix
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
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
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder="/Users/briford/work/sageworks/applications/aws_dashboard/assets",
    )

    app.layout = html.Div([component])
    component.children = markdown

    if __name__ == "__main__":
        app.run_server(debug=True)
