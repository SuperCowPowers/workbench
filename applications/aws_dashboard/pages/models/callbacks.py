"""Callbacks for the Model Subpage Web User Interface"""
import logging
from dash import Dash, callback, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# SageWorks Imports
from sageworks.web_components import (
    table,
    model_plot,
)
from sageworks.utils.pandas_utils import deserialize_aws_broker_data
from sageworks.api.model import Model

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


# Updates the model plot when a model row is selected
def update_model_plot_component(app: Dash):
    @app.callback(
        Output("model_plot", "figure"),
        Input("model_details-dropdown", "value"),
        [State("models_table", "data"), State("models_table", "derived_viewport_selected_row_ids")],
        prevent_initial_call=True,
    )
    def generate_model_plot_figure(inference_run, table_data, selected_rows):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        model_uuid = selected_row_data["uuid"]
        m = Model(model_uuid, legacy=True)

        # Model Details Markdown component
        model_plot_fig = model_plot.ModelPlot().update_contents(m, inference_run)

        # Return the details/markdown for these data details
        return model_plot_fig


# Updates the plugin components when a model row is selected
def update_plugins(plugins):
    # Construct a list of Output objects dynamically based on the plugins' slots
    outputs = [Output(component_id, property)
               for plugin in plugins
               for component_id, property in plugin.slots.items()]

    @callback(
        outputs,
        [Input("model_details-dropdown", "value"),
         Input("models_table", "derived_viewport_selected_row_ids")],
        [State("models_table", "data")],
        prevent_initial_call=True,
    )
    def update_plugin_contents(inference_run, selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            raise PreventUpdate

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        model_uuid = selected_row_data["uuid"]

        # Instantiate the Model
        model = Model(model_uuid, legacy=True)

        # Update the plugins and collect the updated properties for each slot
        updated_properties = []
        for plugin in plugins:
            log.important(f"Updating Plugin: {plugin} with Model: {model_uuid} and Inference Run: {inference_run}")
            updated_contents = plugin.update_contents(model, inference_run=inference_run)

            # Assume that the length of contents matches the number of slots for the plugin
            if len(updated_contents) != len(plugin.slots):
                raise ValueError(f"Plugin {plugin} returned {len(updated_contents)} values, but has {len(plugin.slots)} slots.")

            # Append each value from contents to the updated_properties list
            updated_properties.extend(updated_contents)

        # Return the updated properties for each slot
        return updated_properties
