"""Callbacks for the Plugin Subpage Web User Interface"""
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.model_web_view import ModelWebView
from sageworks.web_components import table, model_metrics, model_markdown
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
        if not selected_rows or len(selected_rows) < 2 or len(selected_rows) > 2:
            return no_update
        print(f"NEW PRINT ROWS: {selected_rows}")
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

# Updates the model metrics when a model row is selected
#TODO Add abbreviated model details markdown
def update_model_metrics_components(app: Dash, model_web_view: ModelWebView):
    @app.callback(
        [
            Output("model_metrics_1", "figure"),
            Output("model_metrics_2", "figure")
        ],
        Input("plugin_table", "derived_viewport_selected_row_ids"),
        State("plugin_table", "data"),
        prevent_initial_call=True,
    )
    def generate_model_metrics_figures(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or len(selected_rows) < 2 or len(selected_rows) > 2:
            return no_update
        
        print(f"Metrics Print Rows: {selected_rows}")
        
        # Metrics 1
        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        model_uuid = selected_row_data["uuid"]
        print(f"Model UUID: {model_uuid}")
        model_details = model_web_view.model_details(model_uuid)
        model_metrics_figure = model_metrics.ModelMetrics().generate_component_figure(model_details)

        # Metrics 2
        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[1]]
        model_uuid_2 = selected_row_data["uuid"]
        print(f"Model UUID: {model_uuid_2}")
        model_details_2 = model_web_view.model_details(model_uuid_2)
        model_metrics_figure_2 = model_metrics.ModelMetrics().generate_component_figure(model_details_2)

        # Return the metrics figures
        return [model_metrics_figure, model_metrics_figure_2]