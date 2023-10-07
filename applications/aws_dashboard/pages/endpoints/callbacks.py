"""Callbacks for the Endpoints Subpage Web User Interface"""
from dash import Dash, no_update
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.endpoint_web_view import EndpointWebView
from sageworks.web_components import table, model_markdown
from sageworks.utils.pandas_utils import deserialize_aws_broker_data


def update_endpoints_table(app: Dash):
    @app.callback(
        [
            Output("endpoints_table", "columns"),
            Output("endpoints_table", "data"),
        ],
        Input("aws-broker-data", "data"),
    )
    def endpoints_update(serialized_aws_broker_data):
        """Return the table data for the Endpoints Table"""
        aws_broker_data = deserialize_aws_broker_data(serialized_aws_broker_data)
        endpoints = aws_broker_data["ENDPOINTS"]
        endpoints["id"] = range(len(endpoints))
        column_setup_list = table.Table().column_setup(endpoints, markdown_columns=["Name"])
        return [column_setup_list, endpoints.to_dict("records")]


# Highlights the selected row in the table
def table_row_select(app: Dash, table_name: str):
    @app.callback(
        Output(table_name, "style_data_conditional"),
        Input(table_name, "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def style_selected_rows(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return no_update
        row_style = [
            {
                "if": {"filter_query": "{{id}} ={}".format(i)},
                "backgroundColor": "rgb(80, 80, 80)",
            }
            for i in selected_rows
        ]
        return row_style


# Updates the endpoint details when a endpoint row is selected
def update_endpoint_details(app: Dash, endpoint_web_view: EndpointWebView):
    @app.callback(
        [
            Output("endpoint_details_header", "children"),
            Output("endpoint_details", "children"),
        ],
        Input("endpoints_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return no_update
        print("Calling Model Details...")
        endpoint_details = endpoint_web_view.endpoint_details(selected_rows[0])
        endpoint_details_markdown = model_markdown.ModelMarkdown().generate_markdown(endpoint_details)

        # Name of the data source for the Header
        endpoint_name = endpoint_web_view.endpoint_name(selected_rows[0])
        header = f"Details: {endpoint_name}"

        # Return the details/markdown for these data details
        return [header, endpoint_details_markdown]


# Updates the plugin component when a endpoint row is selected
def update_plugin(app: Dash, plugin, endpoint_web_view: EndpointWebView):
    @app.callback(
        Output(plugin.component_id(), "figure"),
        Input("endpoints_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def update_callback(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return no_update

        print("Calling Model Details...")
        endpoint_details = endpoint_web_view.endpoint_details(selected_rows[0])
        return plugin.generate_component_figure(endpoint_details)
