"""Callbacks/Connections for the Main/Front Dashboard Page"""

from datetime import datetime
from dash import callback, Input, Output

# SageWorks Imports
from sageworks.web_interface.page_views.main_page import MainPage
from sageworks.web_interface.components import table


# Update the last updated time
def last_updated():
    @callback(
        Output("data-last-updated", "children"),
        Input("main_page_refresh", "n_intervals"),
    )
    def refresh_last_updated_time(_n):
        # A string of the new time (in the local time zone)
        return datetime.now().strftime("Last Updated: %Y-%m-%d (%I:%M %p)")


# Update the incoming data table
def incoming_data_update(main_page: MainPage):
    @callback(
        [
            Output("INCOMING_DATA", "columns"),
            Output("INCOMING_DATA", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _incoming_data_update(_n):
        incoming_data = main_page.incoming_data_summary()
        column_setup_list = table.Table().column_setup(incoming_data, markdown_columns=["Name"])
        return [column_setup_list, incoming_data.to_dict("records")]


# Update the ETL Jobs table
def etl_jobs_update(main_page: MainPage):
    @callback(
        [
            Output("GLUE_JOBS", "columns"),
            Output("GLUE_JOBS", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _etl_jobs_update(_n):
        glue_jobs = main_page.glue_jobs_summary()
        column_setup_list = table.Table().column_setup(glue_jobs, markdown_columns=["Name"])
        return [column_setup_list, glue_jobs.to_dict("records")]


# Update the data sources table
def data_sources_update(main_page: MainPage):
    @callback(
        [
            Output("DATA_SOURCES", "columns"),
            Output("DATA_SOURCES", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _data_sources_update(_n):
        data_sources = main_page.data_sources_summary()
        column_setup_list = table.Table().column_setup(data_sources, markdown_columns=["Name"])
        return [column_setup_list, data_sources.to_dict("records")]


# Update the feature sets table
def feature_sets_update(main_page: MainPage):
    @callback(
        [
            Output("FEATURE_SETS", "columns"),
            Output("FEATURE_SETS", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _feature_sets_update(_n):
        feature_sets = main_page.feature_sets_summary()
        column_setup_list = table.Table().column_setup(feature_sets, markdown_columns=["Feature Group"])
        return [column_setup_list, feature_sets.to_dict("records")]


# Update the models table
def models_update(main_page: MainPage):
    @callback(
        [
            Output("MODELS", "columns"),
            Output("MODELS", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _models_update(_n):
        models = main_page.models_summary()
        column_setup_list = table.Table().column_setup(models, markdown_columns=["Model Group"])
        return [column_setup_list, models.to_dict("records")]


# Update the endpoints table
def endpoints_update(main_page: MainPage):
    @callback(
        [
            Output("ENDPOINTS", "columns"),
            Output("ENDPOINTS", "data"),
        ],
        Input("main_page_refresh", "n_intervals"),
    )
    def _endpoints_update(_n):
        endpoints = main_page.endpoints_summary()
        column_setup_list = table.Table().column_setup(endpoints, markdown_columns=["Name"])
        return [column_setup_list, endpoints.to_dict("records")]
