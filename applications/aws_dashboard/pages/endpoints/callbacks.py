"""Callbacks for the Endpoints Subpage Web User Interface"""

import logging
from dash import callback, no_update, Input, Output, State
from dash.exceptions import PreventUpdate
from urllib.parse import urlparse, parse_qs

# Workbench Imports
from workbench.web_interface.page_views.endpoints_page_view import EndpointsPageView
from workbench.web_interface.components import endpoint_metric_plots
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.cached.cached_endpoint import CachedEndpoint

# Get the Workbench logger
log = logging.getLogger("workbench")


def on_page_load():
    @callback(
        Output("endpoints_table", "selectedRows"),
        Output("endpoints_page_loaded", "data"),
        Input("url", "href"),
        Input("endpoints_table", "rowData"),
        State("endpoints_page_loaded", "data"),
        prevent_initial_call=True,
    )
    def _on_page_load(href, row_data, page_already_loaded):
        if page_already_loaded:
            raise PreventUpdate

        if not href or not row_data:
            raise PreventUpdate

        parsed = urlparse(href)
        if parsed.path != "/endpoints":
            raise PreventUpdate

        selected_uuid = parse_qs(parsed.query).get("uuid", [None])[0]
        if not selected_uuid:
            return [row_data[0]], True

        for row in row_data:
            if row.get("uuid") == selected_uuid:
                return [row], True

        raise PreventUpdate


def endpoint_table_refresh(page_view: EndpointsPageView, table: AGTable):
    @callback(
        [Output(component_id, prop) for component_id, prop in table.properties],
        Input("endpoints_refresh", "n_intervals"),
    )
    def _endpoint_table_refresh(_n):
        """Return the table data for the Endpoints Table"""
        page_view.refresh()
        endpoints = page_view.endpoints()
        endpoints["uuid"] = endpoints["Name"]
        endpoints["id"] = range(len(endpoints))
        return table.update_properties(endpoints)


# Updates the endpoint details when a endpoint row is selected
def update_endpoint_metrics(page_view: EndpointsPageView):
    @callback(
        Output("endpoint_metrics", "figure"),
        Input("endpoints_table", "selectedRows"),
        prevent_initial_call=True,
    )
    def generate_endpoint_details_figures(selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        endpoint_uuid = selected_row_data["uuid"]
        print(f"Endpoint UUID: {endpoint_uuid}")

        # Endpoint Details
        endpoint_details = page_view.endpoint_details(endpoint_uuid)

        # Endpoint Metrics
        endpoint_metrics_figure = endpoint_metric_plots.EndpointMetricPlots().update_properties(endpoint_details)

        # Return the details/markdown for these data details
        return endpoint_metrics_figure


# Set up the plugin callbacks that take an endpoint
def setup_plugin_callbacks(plugins):

    # First we'll register internal callbacks for the plugins
    for plugin in plugins:
        plugin.register_internal_callbacks()

    # Now we'll set up the plugin callbacks for their main inputs (endpoints in this case)
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        Input("endpoints_table", "selectedRows"),
    )
    def update_all_plugin_properties(selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        object_uuid = selected_row_data["uuid"]

        # Create the Endpoint object
        endpoint = CachedEndpoint(object_uuid)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(endpoint))

        # Return all the updated properties
        return all_props
