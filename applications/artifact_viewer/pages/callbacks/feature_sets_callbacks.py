"""FeatureSets Callbacks: Callback within the FeatureSets Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary

# Cheese Sauce
all_data = None


def update_last_updated(app: Dash, sageworks_artifacts: ArtifactsSummary):
    @app.callback(Output("last-updated-feature-sets", "children"), Input("feature-set-updater", "n_intervals"))
    def time_updated(n):
        global all_data
        print("Calling Artifact Refresh...")
        sageworks_artifacts.refresh()
        all_data = sageworks_artifacts.view_data()
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_feature_sets_table(app: Dash):
    @app.callback(Output("FEATURE_SETS_DETAILS", "data"), Input("feature-set-updater", "n_intervals"))
    def data_sources_update(n):
        data_sources = all_data["FEATURE_SETS"]
        return data_sources.to_dict("records")
