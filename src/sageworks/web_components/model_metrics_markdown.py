"""A Markdown Component for model metrics"""

from dash import dcc

# SageWorks Imports
from sageworks.api import Model
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.utils.symbols import health_icons


class ModelMetricsMarkdown(ComponentInterface):
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

    def generate_markdown(self, model: Model, inference_run: str) -> str:
        """Create the Markdown for the details/information about the DataSource or the FeatureSet
        Args:
            model (Model): Sageworks Model object
            inference_run (str): Valid capture_uuid
        Returns:
            str: A Markdown string
        """

        # Get model details
        model_details = model.details()

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

        # Model Metrics
        markdown = "### Model Metrics  \n"
        meta_df = model.inference_metadata(inference_run)
        if meta_df is None:
            test_data = "AWS Training Capture"
            test_data_hash = " N/A "
            test_rows = " - "
            description = " - "
        else:
            inference_meta = meta_df.to_dict(orient="records")[0]
            test_data = inference_meta.get("name", " - ")
            test_data_hash = inference_meta.get("data_hash", " - ")
            test_rows = inference_meta.get("num_rows", " - ")
            description = inference_meta.get("description", " - ")

        # Add the markdown for the model test metrics
        markdown += f"**Test Data:** {test_data}  \n"
        markdown += f"**Data Hash:** {test_data_hash}  \n"
        markdown += f"**Test Rows:** {test_rows}  \n"
        markdown += f"**Description:** {description}  \n"

        # Grab the Metrics from the model details
        metrics = model.performance_metrics(capture_uuid=inference_run)
        if metrics is None:
            markdown += "  \nNo Data  \n"
        else:
            markdown += "  \n"
            metrics = metrics.round(3)
            markdown += metrics.to_markdown(index=False)

        return markdown

    @staticmethod
    def _health_tag_markdown(health_tags: list[str]) -> str:
        """Internal method to generate the health tag markdown
        Args:
            health_tags (list[str]): A list of health tags
        Returns:
            str: A markdown string
        """
        # If we have no health tags, then add a bullet for healthy
        markdown = "**Health Checks**\n"  # Header for Health Checks

        # If we have no health tags, then add a bullet for healthy
        if not health_tags:
            markdown += f"* Healthy: {health_icons.get('healthy')}\n\n"
            return markdown

        # Special case for no_activity with no other tags
        if len(health_tags) == 1 and health_tags[0] == "no_activity":
            markdown += f"* Healthy: {health_icons.get('healthy')}\n"
            markdown += f"* No Activity: {health_icons.get('no_activity')}\n\n"
            return markdown

        # If we have health tags, then add a bullet for each tag
        markdown += "\n".join(f"* {tag}: {health_icons.get(tag, '')}" for tag in health_tags)
        markdown += "\n\n"  # Add newlines for separation
        return markdown


if __name__ == "__main__":
    # This class takes in model metrics and generates a Markdown Component
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    from sageworks.api import Model

    # Create the class and get the AWS FeatureSet details
    m = Model("wine-classification")
    inference_run = "training_holdout"

    # Instantiate the DataDetailsMarkdown class
    ddm = ModelMetricsMarkdown()
    component = ddm.create_component("model_metrics_markdown")

    # Generate the markdown
    markdown = ddm.generate_markdown(m, inference_run)

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder="/home/kolmar/sageworks/applications/aws_dashboard/assets",
    )

    app.layout = html.Div([component])
    component.children = markdown

    if __name__ == "__main__":
        app.run_server(debug=True)
