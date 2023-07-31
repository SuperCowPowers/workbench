"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash, no_update, callback_context
from dash.dependencies import Input, Output, State

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.artifacts.artifact import Artifact

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
        Output("INCOMING_DATA", "data"),
        Input("main-updater", "n_intervals"),
    )
    def incoming_data_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        incoming_data = all_data["INCOMING_DATA"]
        return incoming_data.to_dict("records")

    @app.callback(Output("DATA_SOURCES", "data"), Input("main-updater", "n_intervals"))
    def data_sources_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        data_sources = all_data["DATA_SOURCES"]
        data_sources["del"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return data_sources.to_dict("records")

    @app.callback(Output("FEATURE_SETS", "data"), Input("main-updater", "n_intervals"))
    def feature_sets_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        feature_sets = all_data["FEATURE_SETS"]
        feature_sets["del"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return feature_sets.to_dict("records")

    @app.callback(Output("MODELS", "data"), Input("main-updater", "n_intervals"))
    def models_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        models = all_data["MODELS"]
        models["del"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return models.to_dict("records")

    @app.callback(Output("ENDPOINTS", "data"), Input("main-updater", "n_intervals"))
    def endpoints_update(n):
        if all_data is None:
            return [{}]  # Return an empty row
        endpoints = all_data["ENDPOINTS"]
        endpoints["del"] = "<img src='../assets/trash.png' id='trash-icon'>"
        return endpoints.to_dict("records")


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


"""
TODO: See https://dash.plotly.com/dash-core-components/confirmdialogprovider
      for a dialog when deleting artifacts
"""

"""
Storage Code
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
        State("DATA_SOURCES", "data"),
        State("FEATURE_SETS", "data"),
        State("MODELS", "data"),
        State("ENDPOINTS", "data"),
    )
    def show_modal_and_call_remove_callback(
        data_sources_active_cell,
        feature_sets_active_cell,
        models_active_cell,
        endpoints_active_cell,
        no_button_clicks,
        yes_button_clicks,
        is_open,
        modal_body,
        modal_trigger_state_store,
        data_source_data,
        feature_set_data,
        model_data,
        endpoint_data,
    ):
        trigger_input = callback_context.triggered_id
        if trigger_input is None:
            return no_update, no_update, no_update, no_update
        tables = {
            "DATA_SOURCES": [data_sources_active_cell, data_source_data],
            "FEATURE_SETS": [feature_sets_active_cell, feature_set_data],
            "MODELS": [models_active_cell, model_data],
            "ENDPOINTS": [endpoints_active_cell, endpoint_data],
        }
        if trigger_input == "no-button":
            return (
                False,
                no_update,
                "",
                {
                    "action": "no",
                    "table_name": modal_trigger_state_store.get("table_name"),
                    "table_row": modal_trigger_state_store.get("table_row"),
                },
            )
        if trigger_input == "yes-button" and modal_trigger_state_store.get("table_name") in tables:
            return (
                False,
                no_update,
                "",
                {
                    "action": "yes",
                    "table_name": modal_trigger_state_store.get("table_name"),
                    "table_row": modal_trigger_state_store.get("table_row"),
                },
            )
        if (
            trigger_input in tables
            and isinstance(tables[trigger_input][0], dict)
            and tables[trigger_input][0].get("column_id") == "del"
        ):
            table_name = trigger_input.replace("_", " ").title()
            table_row = tables[trigger_input][0]["row"]
            artifact_uuid = tables[trigger_input][1][int(table_row)]["uuid"]
            modal_body = f'Are you sure you want to remove "{artifact_uuid}" from {table_name}?'
            modal_trigger_state_store = {
                "table_name": trigger_input,
                "table_row": table_row,
            }
            return True, modal_body, modal_trigger_state_store, no_update
        return no_update, no_update, no_update, no_update


def remove_artifact(data, row, Artifact: Artifact):
    artifact = Artifact(data[int(row)]["uuid"])
    artifact.delete()
    del data[int(row)]

    # Now we need to do a force_refresh on the SageWorks Artifacts
    # FIXME: TODO
    # update_last_updated(app: Dash, sageworks_artifacts: ArtifactsWebView, force_refresh = False):
    return data
"""
