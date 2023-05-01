"""Callbacks/Connections in the Web User Interface"""
import dash
from dash import Dash
from dash.dependencies import Input, Output
from datetime import datetime

# SageWorks Imports
from sageworks.web_components.model_data import ModelData
from sageworks.web_components import feature_importance, confusion_matrix, model_details, feature_details
from sageworks.views.web_artifacts_summary import WebArtifactsSummary


def update_last_updated(app: Dash):
    @app.callback(Output("last-updated-models", "children"), Input("models-updater", "n_intervals"))
    def time_updated(n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_models_table(app: Dash, sageworks_artifacts: WebArtifactsSummary):
    @app.callback(Output("models_table", "data"), Input("models-updater", "n_intervals"))
    def data_sources_update(n):
        print("Calling Models Refresh...")
        sageworks_artifacts.refresh()
        models = sageworks_artifacts.models_summary()
        return models.to_dict("records")


# Highlights the selected row in the table
def table_row_select(app: Dash, table_name: str):
    @app.callback(
        Output(table_name, "style_data_conditional"),
        Input(table_name, "derived_viewport_selected_row_ids"),
    )
    def style_selected_rows(selected_rows):
        print(f"Selected Rows: {selected_rows}")
        if not selected_rows or selected_rows[0] is None:
            return dash.no_update
        foo = [
            {
                "if": {"filter_query": "{{id}} ={}".format(i)},
                "backgroundColor": "rgb(80, 80, 80)",
            }
            for i in selected_rows
        ]
        return foo


# Updates the feature importance and confusion matrix figures when a model is selected
def update_figures(app: Dash, model_data: ModelData):
    @app.callback(
        [Output("feature_importance", "figure"), Output("confusion_matrix", "figure")],
        Input("models_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_figures(selected_rows):
        print(f"Selected Rows: {selected_rows}")

        # If there's no selection we're going to return figures for the first row (0)
        if not selected_rows or selected_rows[0] is None:
            selected_rows = [0]

        # Grab the data for this row
        model_row_index = selected_rows[0]

        # Generate a figure for the feature importance component
        feature_info = model_data.get_model_feature_importance(model_row_index)
        feature_figure = feature_importance.create_figure(feature_info)

        # Generate a figure for the confusion matrix component
        c_matrix = model_data.get_model_confusion_matrix(model_row_index)
        matrix_figure = confusion_matrix.create_figure(c_matrix)

        # Now return both of the new figures
        return [feature_figure, matrix_figure]


# Updates the model details when a model row is selected
def update_model_details(app: Dash, model_data: ModelData):
    @app.callback(
        Output("model_details", "children"),
        Input("models_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")

        # If there's no selection we're going to return the model details for the first row (0)
        if not selected_rows or selected_rows[0] is None:
            selected_rows = [0]

        # Grab the data for this row
        model_row_index = selected_rows[0]

        # Generate new Details (Markdown) for the selected model
        model_info = model_data.get_model_details(model_row_index)
        model_markdown = model_details.create_markdown(model_info)

        # Return the details/markdown for this model
        return model_markdown


# Updates the feature details when a model row is selected
def update_feature_details(app: Dash, model_data: ModelData):
    @app.callback(
        Output("feature_details", "children"),
        Input("models_table", "derived_viewport_selected_row_ids"),
    )
    def generate_new_markdown(selected_rows):
        print(f"Selected Rows: {selected_rows}")

        # If there's no selection we're going to return the feature details for the first row (0)
        if not selected_rows or selected_rows[0] is None:
            selected_rows = [0]

        # Grab the data for this row
        model_row_index = selected_rows[0]

        # Generate new Details (Markdown) for the features for this model
        feature_info = model_data.get_model_feature_importance(model_row_index)
        feature_markdown = feature_details.create_markdown(feature_info)

        # Return the details/markdown for these features
        return feature_markdown
