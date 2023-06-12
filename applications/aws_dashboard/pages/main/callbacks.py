"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash, no_update, callback_context
from dash.dependencies import Input, Output, State
import json

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
    

def update_artifact_tables(app: Dash):
    @app.callback(
        Output("INCOMING_DATA", "data"), 
        Input("main-updater", "n_intervals"),
    )
    def incoming_data_update(n):
        if all_data is None:
            return {}
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update
        incoming_data = all_data["INCOMING_DATA"]
        incoming_data["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return incoming_data.to_dict("records")

    @app.callback(
        Output("DATA_SOURCES", "data"),
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("DATA_SOURCES", "data"),
        prevent_initial_call=True
    )
    def data_sources_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update
        if trigger_input == "remove-artifact-store" and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "DATA_SOURCES":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.data_sources.data_source import DataSource
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, DataSource)
                return updated_data
            elif remove_artifact_store.get("action") == "no":
                return no_update
            else:
                return no_update
            
        data_sources = all_data["DATA_SOURCES"]
        data_sources["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return data_sources.to_dict("records")

    @app.callback(
        Output("FEATURE_SETS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("FEATURE_SETS", "data"),
        prevent_initial_call=True
    )
    def feature_sets_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update
        if trigger_input == "remove-artifact-store" and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "FEATURE_SETS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.feature_sets.feature_set import FeatureSet
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, FeatureSet)
                return updated_data
            elif remove_artifact_store.get("action") == "no":
                return no_update
            else:
                return no_update
        feature_sets = all_data["FEATURE_SETS"]
        feature_sets["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return feature_sets.to_dict("records")

    @app.callback(
        Output("MODELS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("MODELS", "data"),
        prevent_initial_call=True
    )
    def models_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update
        if trigger_input == "remove-artifact-store" and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "MODELS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.models.model import Model
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, Model)
                return updated_data
            elif remove_artifact_store.get("action") == "no":
                return no_update
            else:
                return no_update 
        models = all_data["MODELS"]
        models["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return models.to_dict("records")

    @app.callback(
        Output("ENDPOINTS", "data"), 
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("ENDPOINTS", "data"),
        prevent_initial_call=True
    )
    def endpoints_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update
        if trigger_input == "remove-artifact-store" and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "ENDPOINTS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.endpoints.endpoint import Endpoint
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, Endpoint)
                return updated_data
            elif remove_artifact_store.get("action") == "no":
                return no_update
            else:
                return no_update
        
        endpoints = all_data["ENDPOINTS"]
        endpoints["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return endpoints.to_dict("records")


def remove_artifact_callbacks(app: Dash):
    @app.callback(
        Output("modal", "is_open"),
        Output("modal-body", "children"),
        Output("modal-trigger-state-store", "data"),
        Output("remove-artifact-store", "data"),
        Input("DATA_SOURCES", "active_cell"),
        Input("FEATURE_SETS", "active_cell"),
        Input("MODELS", "active_cell"),
        Input("ENDPOINTS", "active_cell"),
        Input("no-button", "n_clicks"),
        Input("yes-button", "n_clicks"),
        State("modal", "is_open"),
        State("modal-body", "children"),
        State("modal-trigger-state-store", "data"),
        State("remove-artifact-store", "data"),
    )
    def pick_right_modal(data_sources_active_cell, feature_sets_active_cell, models_active_cell, endpoints_active_cell, no_button_clicks, yes_button_clicks, is_open, modal_body, modal_trigger_state_store, remove_artifact_store):
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update, no_update, no_update, no_update
        tables = {
            "DATA_SOURCES": data_sources_active_cell, 
            "FEATURE_SETS": feature_sets_active_cell,
            "MODELS": models_active_cell,
            "ENDPOINTS": endpoints_active_cell
        }
        if trigger_input == "no-button":
            remove_artifact_store = {"action": "no", "table_name": modal_trigger_state_store.get("table_name"), "table_row": modal_trigger_state_store.get("table_row")}
            return False, no_update, "", remove_artifact_store
        if trigger_input == "yes-button" and modal_trigger_state_store.get("table_name") in tables:
            remove_artifact_store = {"action": "yes", "table_name": modal_trigger_state_store.get("table_name"), "table_row": modal_trigger_state_store.get("table_row")}
            return False, no_update, "", remove_artifact_store
        if trigger_input in tables and isinstance(tables[trigger_input], dict) and tables[trigger_input].get("column_id") == "remove":
            modal_body = f"Are you sure you want to remove this row from {trigger_input}?"
            modal_trigger_state_store = {"table_name": trigger_input, "table_row": tables[trigger_input]["row"]}
            return True, modal_body, modal_trigger_state_store, no_update
        return no_update, no_update, no_update, no_update


def remove_artifact(data, row, Artifact: Artifact):
    artifact = Artifact(data[int(row)]['uuid'])
    artifact.delete()
    del data[int(row)]
    return data
    