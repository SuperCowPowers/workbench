"""A Markdown Component for details/information about Endpoints"""

# Dash Imports
from dash import html, callback, no_update, dcc
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.api import Endpoint
from sageworks.web_components.component_interface import ComponentInterface
from sageworks.utils.markdown_utils import health_tag_markdown


class EndpointDetails(ComponentInterface):
    """Model Markdown Component"""

    def __init__(self):
        self.prefix_id = ""
        self.endpoint = None
        super().__init__()

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        self.prefix_id = component_id
        container = html.Div(
            id=self.prefix_id,
            children=[
                html.H3(id=f"{self.prefix_id}-header", children="Endpoint: Loading..."),
                dcc.Markdown(id=f"{self.prefix_id}-details"),
            ],
        )
        return container

    def register_callbacks(self, endpoint_table):
        @callback(
            [
                Output(f"{self.prefix_id}-header", "children"),
                Output(f"{self.prefix_id}-details", "children"),
            ],
            Input(endpoint_table, "derived_viewport_selected_row_ids"),
            State(endpoint_table, "data"),
        )
        def update_endpoint(selected_rows, table_data):
            # Check for no selected rows
            if not selected_rows or selected_rows[0] is None:
                return no_update

            # Get the selected row data, grab the uuid, and set the Model object
            selected_row_data = table_data[selected_rows[0]]
            endpoint_uuid = selected_row_data["uuid"]
            self.endpoint = Endpoint(endpoint_uuid, legacy=True)

            # Update the header, the summary, and the details
            header = f"Model: {self.endpoint.uuid}"
            details = self.endpoint_details()

            return header, details

    def endpoint_details(self):
        """Construct the markdown string for the endpoint details

        Returns:
            str: A markdown string
        """
        # Get these fields from the endpoint
        show_fields = ["health_tags", "input", "status", "instance", "variant"]

        # Construct the markdown string
        summary = self.endpoint.details()
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

            # Chop off the "sageworks_" prefix
            key = key.replace("sageworks_", "")

            # Add to markdown string
            markdown += f"**{key}:** {value}  \n"

        return markdown


if __name__ == "__main__":
    # This class takes in endpoint details and generates a details Markdown component
    import dash
    import dash_bootstrap_components as dbc
    from sageworks.web_components.table import Table
    from sageworks.views.artifacts_web_view import ArtifactsWebView

    # Create a endpoint table
    endpoints_table = Table().create_component(
        "endpoints_table", header_color="rgb(60, 100, 60)", row_select="single", max_height=270
    )

    # Populate the table with data
    view = ArtifactsWebView()
    endpoints = view.endpoints_summary()
    endpoints["id"] = range(len(endpoints))
    column_setup_list = Table().column_setup(endpoints, markdown_columns=["Name"])
    endpoints_table.columns = column_setup_list
    endpoints_table.data = endpoints.to_dict("records")

    # Instantiate the EndpointDetails class
    md = EndpointDetails()
    details_component = md.create_component("endpoint_details")

    # Register the callbacks
    md.register_callbacks("endpoints_table")

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder="/Users/briford/work/sageworks/applications/aws_dashboard/assets",
    )

    app.layout = html.Div([endpoints_table, details_component])
    app.run(debug=True)
