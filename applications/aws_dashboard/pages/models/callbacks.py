"""Callbacks for the Model Subpage Web User Interface"""

import logging
from dash import callback, Input, Output, State
from dash.exceptions import PreventUpdate
from urllib.parse import urlparse, parse_qs

# Workbench Imports
from workbench.web_interface.page_views.models_page_view import ModelsPageView
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.cached.cached_model import CachedModel

# Get the Workbench logger
log = logging.getLogger("workbench")

# Theme store ID (defined in app.py)
THEME_STORE_ID = "workbench-theme-store"


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

        selected_name = parse_qs(parsed.query).get("name", [None])[0]
        if not selected_name:
            return [row_data[0]], True

        for row in row_data:
            if row.get("name") == selected_name:
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
        models["name"] = models["Model Group"]
        models["id"] = range(len(models))
        return table.update_properties(models)


def setup_plugin_callbacks(plugins, model_details_plugin):
    """Set up callbacks for all plugins.

    Args:
        plugins: List of all plugins (model_details, shap_plot, model_plot)
        model_details_plugin: The model_details plugin (handles the inference dropdown)
    """

    # Register internal callbacks for all plugins
    for plugin in plugins:
        plugin.register_internal_callbacks()

    # Separate model_details from other plugins - it sets the dropdown that others depend on
    other_plugins = [p for p in plugins if p is not model_details_plugin]

    # Callback to update model_details when a model is selected (this sets the dropdown)
    @callback(
        [Output(cid, prop) for cid, prop in model_details_plugin.properties],
        Input("models_table", "selectedRows"),
        State("model_details-dropdown", "value"),
        State("model_details-header", "children"),
    )
    def update_model_details(selected_rows, inference_run, previous_model_name):
        """Update model_details when a model is selected."""
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        model_name = selected_rows[0]["name"]
        model = CachedModel(model_name)

        return model_details_plugin.update_properties(
            model, inference_run=inference_run, previous_model_name=previous_model_name
        )

    # Callback to update other plugins when the dropdown changes (triggered by model_details)
    @callback(
        [Output(cid, prop) for p in other_plugins for cid, prop in p.properties],
        Input("model_details-dropdown", "value"),
        State("models_table", "selectedRows"),
    )
    def update_other_plugins(inference_run, selected_rows):
        """Update other plugins when inference run changes."""
        if not selected_rows or selected_rows[0] is None or inference_run is None:
            raise PreventUpdate

        model_name = selected_rows[0]["name"]
        model = CachedModel(model_name)

        all_props = []
        for plugin in other_plugins:
            all_props.extend(plugin.update_properties(model, inference_run=inference_run))

        return all_props


def setup_theme_callback(plugins):
    """Set up a callback to update all plugins when the theme changes.

    Args:
        plugins: List of all plugin instances on this page.
    """

    @callback(
        [Output(cid, prop, allow_duplicate=True) for p in plugins for cid, prop in p.properties],
        Input(THEME_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def on_theme_change(theme):
        """Update all plugins when the theme changes."""
        all_props = []
        for plugin in plugins:
            all_props.extend(plugin.set_theme(theme))
        return all_props
