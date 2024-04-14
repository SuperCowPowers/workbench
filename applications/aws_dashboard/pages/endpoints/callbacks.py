"""Callbacks for the Endpoints Subpage Web User Interface"""

import logging
from dash import Dash, callback, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.endpoint_web_view import EndpointWebView
from sageworks.web_components import table, endpoint_metric_plots
from sageworks.utils.pandas_utils import deserialize_aws_broker_data
from sageworks.api.endpoint import Endpoint

# Get the SageWorks logger
log = logging.getLogger("sageworks")


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
        if not selected_rows or selected_rows[0] is None:
            return no_update
        row_style = [
            {
                "if": {"filter_query": "{{id}}={}".format(i)},
                "backgroundColor": "rgb(80, 80, 80)",
            }
            for i in selected_rows
        ]
        # Style for symbols
        symbol_style = {"if": {"column_id": "Health"}, "fontSize": 16, "textAlign": "left"}

        # Append the symbol style to the row style
        row_style.append(symbol_style)
        return row_style


# Updates the endpoint details when a endpoint row is selected
def update_endpoint_metrics(app: Dash, endpoint_web_view: EndpointWebView):
    @app.callback(
        Output("endpoint_metrics", "figure"),
        Input("endpoints_table", "derived_viewport_selected_row_ids"),
        State("endpoints_table", "data"),
        prevent_initial_call=True,
    )
    def generate_endpoint_details_figures(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        endpoint_uuid = selected_row_data["uuid"]
        print(f"Endpoint UUID: {endpoint_uuid}")

        # Endpoint Details
        endpoint_details = endpoint_web_view.endpoint_details(endpoint_uuid)

        # Endpoint Metrics
        endpoint_metrics_figure = endpoint_metric_plots.EndpointMetricPlots().update_properties(endpoint_details)

        # Return the details/markdown for these data details
        return endpoint_metrics_figure


# Set up the plugin callbacks that take an endpoint
def setup_plugin_callbacks(plugins):
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        Input("endpoints_table", "derived_viewport_selected_row_ids"),
        State("endpoints_table", "data"),
    )
    def update_all_plugin_properties(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        object_uuid = selected_row_data["uuid"]

        # Create the Endpoint object
        endpoint = Endpoint(object_uuid, legacy=True)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(endpoint))

        # Return all the updated properties
        return all_props
