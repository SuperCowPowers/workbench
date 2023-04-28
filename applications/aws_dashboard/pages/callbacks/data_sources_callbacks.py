"""FeatureSets Callbacks: Callback within the DataSources Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.views.data_source_view import DataSourceView


def update_last_updated(app: Dash):
    @app.callback(Output("last-updated-data-sources", "children"), Input("data-sources-updater", "n_intervals"))
    def time_updated(n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_data_sources_table(app: Dash, data_source_view: DataSourceView):
    @app.callback(Output("DATA_SOURCES_DETAILS", "data"), Input("data-sources-updater", "n_intervals"))
    def data_sources_update(n):
        print("Calling DataSources Refresh...")
        data_source_view.refresh()
        data_sources = data_source_view.data_sources_summary()
        return data_sources.to_dict("records")
