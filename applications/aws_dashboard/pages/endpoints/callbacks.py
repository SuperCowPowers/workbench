"""FeatureSets Callbacks: Callback within the FeatureSets Web User Interface"""
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.web_components import table
from sageworks.utils.pandas_utils import deserialize_aws_broker_data


def update_last_updated(app: Dash):
    @app.callback(
        Output("last-updated-endpoints", "children"),
        Input("endpoints-updater", "n_intervals"),
    )
    def time_updated(n):
        return datetime.now().strftime("Last Updated: %Y-%m-%d %H:%M:%S")


def update_endpoints_table(app: Dash):
    @app.callback(
        [
            Output("endpoints_table", "columns"),
            Output("endpoints_table", "data"),
        ],
        Input("aws-broker-data", "data"),
    )
    def endpoints_update(serialized_aws_broker_data):
        aws_broker_data = deserialize_aws_broker_data(serialized_aws_broker_data)
        endpoints = aws_broker_data["ENDPOINTS"]
        endpoints["id"] = range(len(endpoints))
        column_setup_list = table.Table().column_setup(endpoints, markdown_columns=["Name"])
        return [column_setup_list, endpoints.to_dict("records")]
