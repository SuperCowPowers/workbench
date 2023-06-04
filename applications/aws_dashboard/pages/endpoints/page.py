"""Endpoints:  A SageWorks Web Interface to view, interact, and manage Endpoints"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.web_components import line_chart
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.web_components import table

# Local Imports
from .layout import endpoints_layout
from . import callbacks

register_page(
    __name__,
    path="/endpoints",
    name="SageWorks - Endpoints",
)


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsWebView()
endpoints_summary = sageworks_artifacts.endpoints_summary()

# Create a table to display the feature sets
endpoints_table = table.create(
    "ENDPOINTS_DETAILS",
    endpoints_summary,
    header_color="rgb(100, 60, 100)",
    row_select="single",
    markdown_columns=["Name"],
)

# Create a fake scatter plot
endpoint_traffic = line_chart.create()

# Create our components
components = {
    "endpoints_table": endpoints_table,
    "endpoint_traffic": endpoint_traffic,
}

# Set up our layout (Dash looks for a var called layout)
layout = endpoints_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_endpoints_table(app, sageworks_artifacts)
