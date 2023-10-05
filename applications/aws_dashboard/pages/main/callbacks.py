"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.web_components import table

# Cheese Sauce
all_data = None


def update_last_updated(app: Dash, web_view: ArtifactsWebView, force_refresh=False):
    @app.callback(Output("last-updated", "children"), Input("main-updater", "n_intervals"))
    def time_updated(n):
        global all_data
        print("Calling ALL Artifact Refresh...")
        web_view.refresh(force_refresh=force_refresh)
        all_data = web_view.view_data()
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_artifact_tables(app: Dash):
    @app.callback(
        [
            Output("INCOMING_DATA", "columns"),
            Output("INCOMING_DATA", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def incoming_data_update(n):
        # The columns need to be in a special format for the DataTable
        incoming_data = all_data["INCOMING_DATA"]
        column_setup_list = table.Table().column_setup(incoming_data, markdown_columns=["Name"])
        return [column_setup_list, incoming_data.to_dict("records")]

    @app.callback(
        [
            Output("GLUE_JOBS", "columns"),
            Output("GLUE_JOBS", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def incoming_data_update(n):
        # The columns need to be in a special format for the DataTable
        glue_jobs = all_data["GLUE_JOBS"]
        column_setup_list = table.Table().column_setup(glue_jobs, markdown_columns=["Name"])
        return [column_setup_list, glue_jobs.to_dict("records")]

    @app.callback(
        [
            Output("DATA_SOURCES", "columns"),
            Output("DATA_SOURCES", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def data_sources_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        data_sources = all_data["DATA_SOURCES"]
        column_setup_list = table.Table().column_setup(data_sources, markdown_columns=["Name"])
        return [column_setup_list, data_sources.to_dict("records")]

    @app.callback(
        [
            Output("FEATURE_SETS", "columns"),
            Output("FEATURE_SETS", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def feature_sets_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        feature_sets = all_data["FEATURE_SETS"]
        column_setup_list = table.Table().column_setup(feature_sets, markdown_columns=["Feature Group"])
        return [column_setup_list, feature_sets.to_dict("records")]

    @app.callback(
        [
            Output("MODELS", "columns"),
            Output("MODELS", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def models_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        models = all_data["MODELS"]
        column_setup_list = table.Table().column_setup(models, markdown_columns=["Model Group"])
        return [column_setup_list, models.to_dict("records")]

    @app.callback(
        [
            Output("ENDPOINTS", "columns"),
            Output("ENDPOINTS", "data"),
        ],
        Input("main-updater", "n_intervals"),
        prevent_initial_call=True,
    )
    def endpoints_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        endpoints = all_data["ENDPOINTS"]
        column_setup_list = table.Table().column_setup(endpoints, markdown_columns=["Name"])
        return [column_setup_list, endpoints.to_dict("records")]


def delete_artifact_callbacks(app: Dash, web_view: ArtifactsWebView):
    @app.callback(Output("modal", "is_open"), Input("DATA_SOURCES", "active_cell"), State("DATA_SOURCES", "data"))
    def delete_data_source(active_cell, table_data):
        global all_data
        if active_cell is None or active_cell["column_id"] != "del":
            print("Delete Cell not pressed...")
            return no_update

        # Get the UUID of the artifact to remove
        uuid = table_data[active_cell["row"]].get("uuid")
        if uuid:
            print(f"Deleting artifact with UUID: {uuid}...")
            web_view.delete_artifact(uuid)
            web_view.refresh(force_refresh=True)
            all_data = web_view.view_data()
            update_artifact_tables(app)
        return no_update
