"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash, no_update
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.artifact import Artifact

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

def remove_artifact(data, row, Artifact: Artifact):
    artifact = Artifact(data[int(row)]['uuid'])
    artifact.delete()
    del data[int(row)]
    return data

def update_artifact_tables(app: Dash):
    @app.callback(
        Output("INCOMING_DATA", "data"), 
        Input("main-updater", "n_intervals"),
    )
    def incoming_data_update(n):
        if all_data is None:
            return {}
        incoming_data = all_data["INCOMING_DATA"]
        return incoming_data.to_dict("records")

    @app.callback(
        Output("DATA_SOURCES", "data"), 
        Input("main-updater", "n_intervals"),
        Input("DATA_SOURCES", "active_cell"),
        State("DATA_SOURCES", "data")
    )
    def data_sources_update(n, active_cell, data):
        if all_data is None:
            return {}
        # if clicked remove cell, remove row
        if active_cell and active_cell["column_id"]=="remove":
            from sageworks.artifacts.data_sources.data_source import DataSource
            row = active_cell["row"]
            updated_data = remove_artifact(data, row, DataSource)
            return updated_data
        data_sources = all_data["DATA_SOURCES"]
        data_sources["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return data_sources.to_dict("records")

    @app.callback(
        Output("FEATURE_SETS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("FEATURE_SETS", "active_cell"),
        State("FEATURE_SETS", "data")
    )
    def feature_sets_update(n, active_cell, data):
        if all_data is None:
            return {}
        # if clicked remove cell, remove row
        if active_cell and active_cell["column_id"]=="remove":
            from sageworks.artifacts.feature_sets.feature_set import FeatureSet
            row = active_cell["row"]
            updated_data = remove_artifact(data, row, FeatureSet)
            return updated_data
        feature_sets = all_data["FEATURE_SETS"]
        feature_sets["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return feature_sets.to_dict("records")

    @app.callback(
        Output("MODELS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("MODELS", "active_cell"),
        State("MODELS", "data")
    )
    def models_update(n, active_cell, data):
        if all_data is None:
            return {}
        # if clicked remove cell, remove row
        if active_cell and active_cell["column_id"]=="remove":
            from sageworks.artifacts.models.model import Model
            row = active_cell["row"]
            updated_data = remove_artifact(data, row, Model)
            return updated_data
        # otherwise, return all data
        models = all_data["MODELS"]
        models["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
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
        if active_cell and active_cell["column_id"]=="remove":
            from sageworks.artifacts.endpoints.endpoint import Endpoint
            row = active_cell["row"]
            updated_data = remove_artifact(data, row, Endpoint)
            return updated_data
        # otherwise, return all data
        endpoints = all_data["ENDPOINTS"]
        endpoints["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return endpoints.to_dict("records")
    