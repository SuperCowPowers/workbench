"""FeatureSets Callbacks: Callback within the FeatureSets Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.web_artifacts_summary import WebArtifactsSummary


def update_last_updated(app: Dash):
    @app.callback(Output("last-updated-feature-sets", "children"), Input("feature-sets-updater", "n_intervals"))
    def time_updated(n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_feature_sets_table(app: Dash, sageworks_artifacts: WebArtifactsSummary):
    @app.callback(Output("FEATURE_SETS_DETAILS", "data"), Input("feature-sets-updater", "n_intervals"))
    def data_sources_update(n):
        print("Calling FeatureSets Refresh...")
        sageworks_artifacts.refresh()
        feature_sets = sageworks_artifacts.feature_sets_summary()
        return feature_sets.to_dict("records")
