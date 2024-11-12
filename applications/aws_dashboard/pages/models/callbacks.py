"""Callbacks for the Model Subpage Web User Interface"""

import logging
from dash import Dash, callback, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.web_components import table, model_plot
from sageworks.utils.pandas_utils import deserialize_aws_broker_data
from sageworks.cached.cached_model import CachedModel

# Get the SageWorks logger
log = logging.getLogger("sageworks")


def update_models_table(app: Dash):
    @app.callback(
        [Output("models_table", "columns"), Output("models_table", "data")],
        Input("aws-broker-data", "data"),
    )
    def models_update(serialized_aws_broker_data):
        """Return the table data for the Models Table"""
        aws_broker_data = deserialize_aws_broker_data(serialized_aws_broker_data)
        models = aws_broker_data["MODELS"]
        models["id"] = range(len(models))
        column_setup_list = table.Table().column_setup(models, markdown_columns=["Model Group"])
        return [column_setup_list, models.to_dict("records")]


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


# Updates the model plot when the model inference run is changed
def update_model_plot_component(app: Dash):
    @app.callback(
        Output("model_plot", "figure"),
        Input("model_details-dropdown", "value"),
        State("models_table", "data"),
        State("models_table", "derived_viewport_selected_row_ids"),
        prevent_initial_call=True,
    )
    def generate_model_plot_figure(inference_run, table_data, selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
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
        Input("models_table", "derived_viewport_selected_row_ids"),
        State("models_table", "data"),
    )
    def update_all_plugin_properties(inference_run, selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        object_uuid = selected_row_data["uuid"]

        # Create the Model object
        model = CachedModel(object_uuid)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(model, inference_run=inference_run))

        # Return all the updated properties
        return all_props
