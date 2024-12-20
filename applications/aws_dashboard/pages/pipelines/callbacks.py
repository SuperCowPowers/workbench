"""Callbacks for the Pipelines Subpage Dashboard Interface"""

import logging

from dash import callback, Output, Input, State
from dash.exceptions import PreventUpdate
from urllib.parse import urlparse, parse_qs


# Workbench Imports
from workbench.web_interface.page_views.pipelines_page_view import PipelinesPageView
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.cached.cached_pipeline import CachedPipeline

# Get the Workbench logger
log = logging.getLogger("workbench")


def on_page_load():
    @callback(
        Output("pipelines_table", "selectedRows"),
        Output("pipelines_page_loaded", "data"),
        Input("url", "href"),
        Input("pipelines_table", "rowData"),
        State("pipelines_page_loaded", "data"),
        prevent_initial_call=True,
    )
    def _on_page_load(href, row_data, page_already_loaded):
        if page_already_loaded:
            raise PreventUpdate

        if not href or not row_data:
            raise PreventUpdate

        parsed = urlparse(href)
        if parsed.path != "/pipelines":
            raise PreventUpdate

        selected_uuid = parse_qs(parsed.query).get("uuid", [None])[0]
        if not selected_uuid:
            return [row_data[0]], True

        for row in row_data:
            if row.get("uuid") == selected_uuid:
                return [row], True

        raise PreventUpdate


def pipeline_table_refresh(page_view: PipelinesPageView, table: AGTable):
    @callback(
        [Output(component_id, prop) for component_id, prop in table.properties],
        Input("pipelines_refresh", "n_intervals"),
    )
    def _pipeline_table_refresh(_n):
        """Return the table data for the Pipelines Table"""
        page_view.refresh()
        pipelines = page_view.pipelines()
        pipelines["uuid"] = pipelines["Name"]
        pipelines["id"] = range(len(pipelines))
        return table.update_properties(pipelines)


# Set up the plugin callbacks that take a pipeline
def setup_plugin_callbacks(plugins):
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        Input("pipelines_table", "selectedRows"),
    )
    def update_all_plugin_properties(selected_rows):
        # Check for no selected rows
        if not selected_rows:
            raise PreventUpdate

        # Get the selected row data and grab the name
        selected_row_data = selected_rows[0]
        pipeline_name = selected_row_data["Name"]

        # Create the Endpoint object
        pipeline = CachedPipeline(pipeline_name)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(pipeline))

        # Return all the updated properties
        return all_props
