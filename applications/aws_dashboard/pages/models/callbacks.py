"""Callbacks for the Model Subpage Web User Interface"""

import logging
from dash import callback, no_update, Input, Output, State
from dash.exceptions import PreventUpdate
from urllib.parse import urlparse, parse_qs

# Workbench Imports
from workbench.api import ModelType
from workbench.web_interface.page_views.models_page_view import ModelsPageView
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.web_interface.components.plugins.scatter_plot import ScatterPlot
from workbench.web_interface.components.plugins.confusion_matrix import ConfusionMatrix
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


def setup_plugin_callbacks(plugins, scatter_plot_plugin: ScatterPlot, confusion_matrix_plugin: ConfusionMatrix):
    """Set up callbacks for all plugins including model-type-aware visualization switching.

    Args:
        plugins: List of all plugins (model_details, shap_plot, scatter_plot, confusion_matrix)
        scatter_plot_plugin: The ScatterPlot plugin instance (for regression models)
        confusion_matrix_plugin: The ConfusionMatrix plugin instance (for classification models)
    """

    # Register internal callbacks for all plugins
    for plugin in plugins:
        plugin.register_internal_callbacks()

    # Separate plugins by type - ScatterPlot and ConfusionMatrix need special handling
    standard_plugins = [p for p in plugins if p not in [scatter_plot_plugin, confusion_matrix_plugin]]

    # Callback to update standard plugins (model_details, shap_plot)
    @callback(
        [Output(component_id, prop) for p in standard_plugins for component_id, prop in p.properties],
        Input("models_table", "selectedRows"),
        State("model_details-dropdown", "value"),
        State("model_details-header", "children"),  # Current model name from header
    )
    def update_standard_plugin_properties(selected_rows, inference_run, previous_model_name):
        """Update properties for standard plugins that take a Model object."""
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        model_name = selected_rows[0]["name"]
        model = CachedModel(model_name)

        all_props = []
        for p in standard_plugins:
            # Pass current inference_run and previous model name to detect model changes
            all_props.extend(p.update_properties(
                model,
                inference_run=inference_run,
                previous_model_name=previous_model_name
            ))

        return all_props

    # Build outputs list for the model plot callback
    model_plot_outputs = [
        Output("scatter-plot-container", "style"),
        Output("confusion-matrix-container", "style"),
    ]
    model_plot_outputs.extend([Output(cid, prop) for cid, prop in scatter_plot_plugin.properties])
    model_plot_outputs.extend([Output(cid, prop) for cid, prop in confusion_matrix_plugin.properties])

    # Callback to toggle visibility and update ScatterPlot/ConfusionMatrix based on model type
    @callback(
        model_plot_outputs,
        Input("model_details-dropdown", "value"),
        Input("models_table", "selectedRows"),
    )
    def update_model_plot_plugins(inference_run, selected_rows):
        """Update ScatterPlot or ConfusionMatrix based on model type, toggling visibility."""
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        model_name = selected_rows[0]["name"]
        model = CachedModel(model_name)

        # Determine visibility based on model type
        is_classifier = model.model_type == ModelType.CLASSIFIER
        scatter_style = {"display": "none"} if is_classifier else {"display": "block"}
        confusion_style = {"display": "block"} if is_classifier else {"display": "none"}

        if is_classifier:
            # Update ConfusionMatrix, return no_update for ScatterPlot
            cm_props = confusion_matrix_plugin.update_properties(model, inference_run=inference_run)
            scatter_props = [no_update] * len(scatter_plot_plugin.properties)
        else:
            # Update ScatterPlot with regression data
            df = model.get_inference_predictions(inference_run)
            if df is None:
                raise PreventUpdate

            # Get target column (residual is already computed in CachedModel)
            target = model.target()
            if isinstance(target, list):
                target = next((t for t in target if t in inference_run), target[0])

            # Check if "confidence" column exists for coloring
            color_col = "confidence" if "confidence" in df.columns else "prediction"

            scatter_props = scatter_plot_plugin.update_properties(
                df,
                x=target,
                y="prediction",
                color=color_col,
                regression_line=True,
            )
            cm_props = [no_update] * len(confusion_matrix_plugin.properties)

        return [scatter_style, confusion_style] + scatter_props + cm_props
