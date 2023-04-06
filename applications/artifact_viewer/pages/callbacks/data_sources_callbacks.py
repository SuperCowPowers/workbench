"""Callbacks/Connections in the Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary

# Cheese Sauce
all_data = None


def update_last_updated(app: Dash, sageworks_artifacts: ArtifactsSummary):
    @app.callback(Output("last-updated-data-sources", "children"), Input("data-sources-updater", "n_intervals"))
    def time_updated(n):
        global all_data
        print("Calling Artifact Refresh...")
        sageworks_artifacts.refresh()
        all_data = sageworks_artifacts.view_data()
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_data_source_table(app: Dash):
    @app.callback(Output("DATA_SOURCES_DETAILS", "data"), Input("data-sources-updater", "n_intervals"))
    def data_sources_update(n):
        data_sources = all_data["DATA_SOURCES"]
        return data_sources.to_dict("records")
