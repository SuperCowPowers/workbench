"""Callbacks for the Pipelines Subpage Dashboard Interface"""

import logging
from dash import callback
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

# SageWorks Imports
from sageworks.utils.pandas_utils import deserialize_aws_broker_data
from sageworks.api.pipeline import Pipeline

# Get the SageWorks logger
log = logging.getLogger("sageworks")


def update_pipelines_table(table_object):
    @callback(
        [Output(component_id, prop) for component_id, prop in table_object.properties],
        Input("aws-broker-data", "data"),
    )
    def pipelines_update(serialized_aws_broker_data):
        """Return the table data for the Pipelines Table"""
        aws_broker_data = deserialize_aws_broker_data(serialized_aws_broker_data)
        pipelines = aws_broker_data["PIPELINES"]
        return table_object.update_properties(pipelines)


# Set up the plugin callbacks that take a pipeline
def setup_plugin_callbacks(plugins):
    @callback(
        # Aggregate plugin outputs
        [Output(component_id, prop) for p in plugins for component_id, prop in p.properties],
        Input("pipelines_table", "selectedRows"),
    )
    def update_all_plugin_properties(selected_rows):
        # Check for no selected rows
        if not selected_rows:
            raise PreventUpdate

        # Get the selected row data and grab the name
        selected_row_data = selected_rows[0]
        pipeline_name = selected_row_data["name"]

        # Create the Endpoint object
        pipeline = Pipeline(pipeline_name)

        # Update all the properties for each plugin
        all_props = []
        for p in plugins:
            all_props.extend(p.update_properties(pipeline))

        # Return all the updated properties
        return all_props
