"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView

# Cheese Sauce
all_data = None


def update_last_updated(app: Dash, sageworks_artifacts: ArtifactsWebView):
    @app.callback(Output("last-updated", "children"), Input("main-updater", "n_intervals"))
    def time_updated(n):
        global all_data
        print("Calling ALL Artifact Refresh...")
        sageworks_artifacts.refresh()
        all_data = sageworks_artifacts.view_data()
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_artifact_tables(app: Dash):
    @app.callback(Output("INCOMING_DATA", "data"), Input("main-updater", "n_intervals"))
    def incoming_data_update(n):
        if all_data is None:
            return {}
        incoming_data = all_data["INCOMING_DATA"]
        return incoming_data.to_dict("records")

    @app.callback(Output("DATA_SOURCES", "data"), Input("main-updater", "n_intervals"))
    def data_sources_update(n):
        if all_data is None:
            return {}
        data_sources = all_data["DATA_SOURCES"]
        return data_sources.to_dict("records")

    @app.callback(Output("FEATURE_SETS", "data"), Input("main-updater", "n_intervals"))
    def feature_sets_update(n):
        if all_data is None:
            return {}
        feature_sets = all_data["FEATURE_SETS"]
        return feature_sets.to_dict("records")

    @app.callback(Output("MODELS", "data"), Input("main-updater", "n_intervals"))
    def models_update(n):
        if all_data is None:
            return {}
        models = all_data["MODELS"]
        return models.to_dict("records")

    @app.callback(
        Output("ENDPOINTS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("ENDPOINTS", "active_cell"), 
        State("ENDPOINTS", "data")
    )
    def endpoints_update(n, active_cell, data):
        # if no data, return empty dict
        if all_data is None:
            return {}
        # if clicked remove cell, remove row
        if active_cell:
            row = active_cell["row"]
            col = active_cell["column_id"]
            if col == "remove":
                # TODO - need to actually remove Stack resources
                del data[row]
                return data
        # otherwise, return all data
        endpoints = all_data["ENDPOINTS"]
        return endpoints.to_dict("records")