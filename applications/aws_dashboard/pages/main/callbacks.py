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
        incoming_data = all_data["INCOMING_DATA"]
        return incoming_data.to_dict("records")

    @app.callback(
        Output("DATA_SOURCES", "data"),
        Output("remove-artifact-store", "clear_data"),
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("DATA_SOURCES", "data"),
    )
    def data_sources_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        
        trigger_input = callback_context.triggered_id
        if "remove-artifact-store" == trigger_input and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "DATA_SOURCES":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.data_sources.data_source import DataSource
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, DataSource)
                return updated_data, True
            elif remove_artifact_store.get("action") == "no":
                return no_update, True
            else:
                return no_update, False
            
        data_sources = all_data["DATA_SOURCES"]
        data_sources["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return data_sources.to_dict("records"), False

    @app.callback(
        Output("FEATURE_SETS", "data"), 
        Output("remove-artifact-store", "clear_data"),
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("FEATURE_SETS", "data"),
    )
    def feature_sets_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        
        trigger_input = callback_context.triggered_id
        if "remove-artifact-store" == trigger_input and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "FEATURE_SETS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.feature_sets.feature_set import FeatureSet
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, FeatureSet)
                return updated_data, True
            elif remove_artifact_store.get("action") == "no":
                return no_update, True
            else:
                return no_update, False
        
        feature_sets = all_data["FEATURE_SETS"]
        feature_sets["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return feature_sets.to_dict("records"), False

    @app.callback(
        Output("MODELS", "data"), 
        Output("remove-artifact-store", "clear_data"),
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("MODELS", "data")
    )
    def models_update(n, remove_artifact_store, data):
        if all_data is None:
            return {}
        
        trigger_input = callback_context.triggered_id
        if "remove-artifact-store" == trigger_input and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "MODELS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.models.model import Model
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, Model)
                return updated_data, True
            elif remove_artifact_store.get("action") == "no":
                return no_update, True
            else:
                return no_update, False
            
        models = all_data["MODELS"]
        models["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return models.to_dict("records"), False

    @app.callback(
        Output("ENDPOINTS", "data"), 
        Output("remove-artifact-store", "clear_data"),
        Input("main-updater", "n_intervals"),
        Input("remove-artifact-store", "data"),
        State("ENDPOINTS", "data")
    )
    def endpoints_update(n, remove_artifact_store, data):
        # if no data, return empty dict
        if all_data is None:
            return {}
        
        trigger_input = callback_context.triggered_id
        if "remove-artifact-store" == trigger_input and type(remove_artifact_store) == dict and remove_artifact_store.get("table_name") == "ENDPOINTS":
            if remove_artifact_store.get("action") == "yes":
                from sageworks.artifacts.endpoints.endpoint import Endpoint
                row = remove_artifact_store.get("table_row")
                updated_data = remove_artifact(data, row, Endpoint)
                return updated_data, True
            elif remove_artifact_store.get("action") == "no":
                return no_update, True
            else:
                return no_update, False
        
        endpoints = all_data["ENDPOINTS"]
        endpoints["remove"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return endpoints.to_dict("records"), False


def remove_artifact_callbacks(app: Dash):
    @app.callback(
        Output("modal", "is_open"),
        Output("modal-body", "children"),
        Output("modal-trigger-state-store", "data"),
        Output("remove-artifact-store", "data"),
        Output("clear-activate-cell-store", "data"),
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
        tables_names = ["DATA_SOURCES", "FEATURE_SETS", "MODELS", "ENDPOINTS"]
        if trigger_input == "no-button":
            remove_artifact_store = {"action": "no", "table_name": modal_trigger_state_store.get("table_name"), "table_row": modal_trigger_state_store.get("table_row")}
            clear_active_cell = modal_trigger_state_store.get("table_name")
            return False, no_update, "", remove_artifact_store, clear_active_cell
        if trigger_input == "yes-button" and modal_trigger_state_store.get("table_name") in tables_names:
            remove_artifact_store = {"action": "yes", "table_name": modal_trigger_state_store.get("table_name"), "table_row": modal_trigger_state_store.get("table_row")}
            clear_active_cell = modal_trigger_state_store.get("table_name")
            return False, no_update, "", remove_artifact_store, clear_active_cell
        if "DATA_SOURCES" == trigger_input and (data_sources_active_cell != None or data_sources_active_cell != ""):
            return modal_update("DATA_SOURCES", "data source", data_sources_active_cell)
        if "FEATURE_SETS" == trigger_input and (feature_sets_active_cell != None or feature_sets_active_cell != ""): 
            return modal_update("FEATURE_SETS", "feature set", feature_sets_active_cell)
        if "MODELS" == trigger_input and (models_active_cell != None or models_active_cell != ""):
            return modal_update("MODELS", "model", models_active_cell)
        if "ENDPOINTS" == trigger_input and (endpoints_active_cell != None or endpoints_active_cell != ""):
            return modal_update("ENDPOINTS", "endpoint", endpoints_active_cell)
        return no_update, no_update, no_update, no_update, no_update


def clear_active_cell(app: Dash):
    @app.callback(
        Output("DATA_SOURCES", "active_cell"),
        Output("FEATURE_SETS", "active_cell"),
        Output("MODELS", "active_cell"),
        Output("ENDPOINTS", "active_cell"),
        Input("clear-activate-cell-store", "data")
    )
    def _clear_active_cell(clear_activate_cell_store):
        if clear_activate_cell_store != "":
            if clear_activate_cell_store == "DATA_SOURCES":
                return "", no_update, no_update, no_update
            if clear_activate_cell_store == "FEATURE_SETS":
                return no_update, "", no_update, no_update
            if clear_activate_cell_store == "MODELS":
                return no_update, no_update, "", no_update
            if clear_activate_cell_store == "ENDPOINTS":
                return no_update, no_update, no_update, ""
        return no_update, no_update, no_update, no_update
    

def modal_update(table_name, artifact_name, active_cell):
    if active_cell["column_id"]=="remove":
        modal_body = f"Are you sure you want to remove this {artifact_name}?"
        modal_trigger_state_store = {"table_name": table_name, "table_row": active_cell["row"]}
        return True, modal_body, modal_trigger_state_store, no_update, no_update
    return no_update, no_update, no_update, no_update, no_update

def remove_artifact(data, row, Artifact: Artifact):
    artifact = Artifact(data[0][int(row)]['uuid'])
    artifact.delete()
    del data[0][int(row)]
    return data
    