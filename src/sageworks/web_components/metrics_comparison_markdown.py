"""A Markdown Component for details/information about Models"""
import pandas as pd
from dash import dcc

# SageWorks Imports
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.utils.symbols import health_icons


class MetricsComparisonMarkdown(ComponentInterface):
    """Metrics Comparison Markdown Component"""

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

        # Keys
        top_level_details = {
            key: value for key, value in model_details.items() if key in ['uuid', 'input', 'sageworks_tags', 'model_type', 'version', 'description']
        }

        # FIXME: Remove this later: Add the model info to the top level details
        model_info = model_details.get("model_info", {})
        prefixed_model_info = {f"model_{k}": v for k, v in model_info.items()}
        top_level_details.update(prefixed_model_info)

        # Construct the markdown string
        markdown = ""
        for key, value in top_level_details.items():
            # Special case for the health tags
            if key == "health_tags":
                markdown += self._health_tag_markdown(value)
                continue

            # Special case for dataframes
            if isinstance(value, pd.DataFrame):
                value_str = "Dataframe"

            else:
                # Not sure why str() conversion might fail, but we'll catch it
                try:
                    value_str = str(value)[:100]
                except Exception as e:
                    self.log.error(f"Error converting {key} to string: {e}")
                    value_str = "*"

            # Add to markdown string
            markdown += f"**{key}:** {value_str}  \n"
        
        # Grab the Metrics from the model details
        metrics = model_details.get("model_metrics")
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
    # This class takes in model details and generates a Confusion Matrix
    import dash
    from dash import dcc, html, Dash
    import dash_bootstrap_components as dbc
    from sageworks.core.artifacts.model_core import ModelCore

    # Create the class and get the AWS FeatureSet details
    m = ModelCore("abalone-regression")
    model_details = m.details()

    # Instantiate the DataDetailsMarkdown class
    ddm = MetricsComparisonMarkdown()
    component = ddm.create_component("model_markdown")

    # Generate the markdown
    markdown = ddm.generate_markdown(model_details)

    # Initialize Dash app
    app = Dash(
        __name__,
        title="SageWorks Dashboard",
        external_stylesheets=[dbc.themes.DARKLY],
    )

    app.layout = html.Div([component])
    component.children = markdown

    if __name__ == "__main__":
        app.run_server(host="0.0.0.0", port=8000, dev_tools_ui=False, dev_tools_props_check=False, debug=True)
