"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.web_artifacts_summary import WebArtifactsSummary

# Cheese Sauce
all_data = None


def update_last_updated(app: Dash, sageworks_artifacts: WebArtifactsSummary):
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
        incoming_data = all_data["INCOMING_DATA"]
        return incoming_data.to_dict("records")

    @app.callback(Output("DATA_SOURCES", "data"), Input("main-updater", "n_intervals"))
    def data_sources_update(n):
        data_sources = all_data["DATA_SOURCES"]
        return data_sources.to_dict("records")

    @app.callback(Output("FEATURE_SETS", "data"), Input("main-updater", "n_intervals"))
    def feature_sets_update(n):
        feature_sets = all_data["FEATURE_SETS"]
        return feature_sets.to_dict("records")

    @app.callback(Output("MODELS", "data"), Input("main-updater", "n_intervals"))
    def models_update(n):
        models = all_data["MODELS"]
        return models.to_dict("records")

    @app.callback(Output("ENDPOINTS", "data"), Input("main-updater", "n_intervals"))
    def endpoints_update(n):
        endpoints = all_data["ENDPOINTS"]
        return endpoints.to_dict("records")
