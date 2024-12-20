"""Callbacks for the Model Subpage Web User Interface"""

import logging
from dash import callback, no_update, Input, Output, State
from dash.exceptions import PreventUpdate
from urllib.parse import urlparse, parse_qs


# Workbench Imports
from workbench.web_interface.page_views.models_page_view import ModelsPageView
from workbench.web_interface.components import model_plot
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.cached.cached_model import CachedModel

# Get the Workbench logger
log = logging.getLogger("workbench")


def on_page_load():
    @callback(
        Output("models_table", "selectedRows"),
        Output("models_page_loaded", "data"),
        Input("url", "href"),
        Input("models_table", "rowData"),
        State("models_page_loaded", "data"),
        prevent_initial_call=True,
    )
    def _on_page_load(href, row_data, page_already_loaded):
        if page_already_loaded:
            raise PreventUpdate

        if not href or not row_data:
            raise PreventUpdate

        parsed = urlparse(href)
        if parsed.path != "/models":
            raise PreventUpdate

        selected_uuid = parse_qs(parsed.query).get("uuid", [None])[0]
        if not selected_uuid:
            return [row_data[0]], True

        for row in row_data:
            if row.get("uuid") == selected_uuid:
                return [row], True

        raise PreventUpdate


def model_table_refresh(page_view: ModelsPageView, table: AGTable):
    @callback(
        [Output(component_id, prop) for component_id, prop in table.properties],
        Input("models_refresh", "n_intervals"),
    )
    def _model_table_refresh(_n):
        """Return the table data for the Models Table"""
        page_view.refresh()
        models = page_view.models()
        models["uuid"] = models["Model Group"]
        models["id"] = range(len(models))
        return table.update_properties(models)


# Updates the model plot when the model inference run dropdown is changed
def update_model_plot_component():
    @callback(
        Output("model_plot", "figure"),
        Input("model_details-dropdown", "value"),
        Input("models_table", "selectedRows"),
        prevent_initial_call=True,
    )
    def generate_model_plot_figure(inference_run, selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        model_uuid = selected_row_data["uuid"]
        m = CachedModel(model_uuid)

        # Model Details Markdown component
        model_plot_fig = model_plot.ModelPlot().update_properties(m, inference_run)

        # Return the details/markdown for these data details
        return model_plot_fig


def setup_plugin_callbacks(plugins):

    # First we'll register internal callbacks for the plugins
    for plugin in plugins:
        plugin.register_internal_callbacks()

    # Now we'll set up the plugin callbacks for their main inputs (models in this case)
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        State("model_details-dropdown", "value"),
        Input("models_table", "selectedRows"),
    )
    def update_all_plugin_properties(inference_run, selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = selected_rows[0]
        object_uuid = selected_row_data["uuid"]

        # Create the Model object
        model = CachedModel(object_uuid)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(model, inference_run=inference_run))

        # Return all the updated properties
        return all_props
