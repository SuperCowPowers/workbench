"""Callbacks for the Endpoints Subpage Web User Interface"""
import logging
from dash import Dash, callback, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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
        endpoint_metrics_figure = endpoint_metric_plots.EndpointMetricPlots().update_contents(endpoint_details)

        # Return the details/markdown for these data details
        return endpoint_metrics_figure


# Updates the plugin components when a model row is selected
def update_plugins(plugins):
    # Construct a list of Output objects dynamically based on the plugins' slots
    outputs = [Output(component_id, property)
               for plugin in plugins
               for component_id, property in plugin.slots.items()]

    @callback(
        outputs,
        Input("endpoints_table", "derived_viewport_selected_row_ids"),
        State("endpoints_table", "data"),
        prevent_initial_call=True,
    )
    def update_plugin_contents(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        endpoint_uuid = selected_row_data["uuid"]

        # Instantiate the Model
        endpoint = Endpoint(endpoint_uuid, legacy=True)

        # Update the plugins and collect the updated properties for each slot
        updated_properties = []
        for plugin in plugins:
            log.important(f"Updating Plugin: {plugin} with Endpoint: {endpoint_uuid}")
            updated_contents = plugin.update_contents(endpoint)

            # Assume that the length of contents matches the number of slots for the plugin
            if len(updated_contents) != len(plugin.slots):
                raise ValueError(f"Plugin {plugin} returned {len(updated_contents)} values, but has {len(plugin.slots)} slots.")

            # Append each value from contents to the updated_properties list
            updated_properties.extend(updated_contents)

        # Return the updated properties for each slot
        return updated_properties
