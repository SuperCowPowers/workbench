"""Callbacks for the Plugin Subpage Web User Interface"""
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.model_web_view import ModelWebView
from sageworks.web_components import table
from sageworks.utils.pandas_utils import deserialize_aws_broker_data


def update_plugin_table(app: Dash):
    @app.callback(
        [Output("plugin_table", "columns"), Output("plugin_table", "data")],
        Input("aws-broker-data", "data"),
    )
    def models_update(serialized_aws_broker_data):
        """Return the table data for the Plugin/Models Table"""
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
        print(f"Selected Rows: {selected_rows}")
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


# Updates the plugin component when a model row is selected
def update_plugin(app: Dash, plugin, model_web_view: ModelWebView):
    @app.callback(
        Output(plugin.component_id(), "figure"),
        Input("plugin_table", "derived_viewport_selected_row_ids"),
        State("plugin_table", "data"),
        prevent_initial_call=True,
    )
    def update_plugin_figure(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        model_uuid = selected_row_data["uuid"]

        # Get the model details and send it to the plugin
        model_details = model_web_view.model_details(model_uuid)
        return plugin.generate_component_figure(model_details)
