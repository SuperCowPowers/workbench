"""Callbacks for the Model Subpage Web User Interface"""
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.model_web_view import ModelWebView
from sageworks.web_components import table, model_markdown, model_metrics
from sageworks.utils.pandas_utils import deserialize_aws_broker_data


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
        return row_style


# Updates the model details when a model row is selected
def update_model_detail_components(app: Dash, model_web_view: ModelWebView):
    @app.callback(
        [
            Output("model_details_header", "children"),
            Output("model_details", "children"),
            Output("model_metrics", "figure"),
        ],
        Input("models_table", "derived_viewport_selected_row_ids"),
        State("models_table", "data"),
        prevent_initial_call=True,
    )
    def generate_model_details_figures(selected_rows, table_data):
        # Check for no selected rows
        if not selected_rows or selected_rows[0] is None:
            return no_update

        # Get the selected row data and grab the uuid
        selected_row_data = table_data[selected_rows[0]]
        model_uuid = selected_row_data["uuid"]
        print(f"Model UUID: {model_uuid}")

        # Set the Header Text
        header = f"Details: {model_uuid}"

        # Model Details
        model_details = model_web_view.model_details(model_uuid)
        model_details_markdown = model_markdown.ModelMarkdown().generate_markdown(model_details)

        # Model Metrics
        model_metrics_figure = model_metrics.ModelMetrics().generate_component_figure(model_details)

        # Return the details/markdown for these data details
        return [header, model_details_markdown, model_metrics_figure]


# Updates the plugin component when a model row is selected
def update_plugin(app: Dash, plugin, model_web_view: ModelWebView):
    @app.callback(
        Output(plugin.component_id(), "figure"),
        Input("models_table", "derived_viewport_selected_row_ids"),
        State("models_table", "data"),
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
