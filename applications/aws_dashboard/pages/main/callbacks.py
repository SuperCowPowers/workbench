"""Callbacks/Connections in the Web User Interface"""

from datetime import datetime
import hashlib
from dash import Dash, no_update
from dash.dependencies import Input, Output, State


# SageWorks Imports
from sageworks.web_views.artifacts_web_view import ArtifactsWebView
from sageworks.web_components import table
from sageworks.utils.pandas_utils import serialize_aws_metadata, deserialize_aws_metadata


# Helper functions
def content_hash(serialized_data):
    return hashlib.md5(serialized_data.encode()).hexdigest()


def refresh_data(app: Dash, web_view: ArtifactsWebView):
    @app.callback(
        [
            Output("data-last-updated", "children"),
            Output("aws-metadata", "data"),
        ],
        Input("metadata-update-timer", "n_intervals"),
        State("aws-metadata", "data"),
    )
    def refresh_aws_metadata(n, aws_metadata):
        # A string of the new time
        new_time = datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")

        # Grab all the data from the Web View
        new_metadata = web_view.view_data()
        serialized_data = serialize_aws_metadata(new_metadata)

        # Sanity check our existing data
        if aws_metadata is None:
            print("No existing data...")
            new_time = new_time + " (Data Update)"
            return [new_time, serialized_data]

        # If the data hasn't changed just return the new time
        if n and content_hash(aws_metadata) == content_hash(serialized_data):
            return [new_time, no_update]

        # Update both the time and the data
        new_time = new_time + " (Data Update)"
        return [new_time, serialized_data]


def update_artifact_tables(app: Dash):
    @app.callback(
        [
            Output("INCOMING_DATA", "columns"),
            Output("INCOMING_DATA", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def incoming_data_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        incoming_data = aws_metadata["INCOMING_DATA"]
        column_setup_list = table.Table().column_setup(incoming_data, markdown_columns=["Name"])
        return [column_setup_list, incoming_data.to_dict("records")]

    @app.callback(
        [
            Output("GLUE_JOBS", "columns"),
            Output("GLUE_JOBS", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def glue_jobs_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        glue_jobs = aws_metadata["GLUE_JOBS"]
        column_setup_list = table.Table().column_setup(glue_jobs, markdown_columns=["Name"])
        return [column_setup_list, glue_jobs.to_dict("records")]

    @app.callback(
        [
            Output("DATA_SOURCES", "columns"),
            Output("DATA_SOURCES", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def data_sources_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        data_sources = aws_metadata["DATA_SOURCES"]
        column_setup_list = table.Table().column_setup(data_sources, markdown_columns=["Name"])
        return [column_setup_list, data_sources.to_dict("records")]

    @app.callback(
        [
            Output("FEATURE_SETS", "columns"),
            Output("FEATURE_SETS", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def feature_sets_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        feature_sets = aws_metadata["FEATURE_SETS"]
        column_setup_list = table.Table().column_setup(feature_sets, markdown_columns=["Feature Group"])
        return [column_setup_list, feature_sets.to_dict("records")]

    @app.callback(
        [
            Output("MODELS", "columns"),
            Output("MODELS", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def models_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        models = aws_metadata["MODELS"]
        column_setup_list = table.Table().column_setup(models, markdown_columns=["Model Group"])
        return [column_setup_list, models.to_dict("records")]

    @app.callback(
        [
            Output("ENDPOINTS", "columns"),
            Output("ENDPOINTS", "data"),
        ],
        Input("aws-metadata", "data"),
    )
    def endpoints_update(serialized_aws_metadata):
        aws_metadata = deserialize_aws_metadata(serialized_aws_metadata)
        endpoints = aws_metadata["ENDPOINTS"]
        column_setup_list = table.Table().column_setup(endpoints, markdown_columns=["Name"])
        return [column_setup_list, endpoints.to_dict("records")]
