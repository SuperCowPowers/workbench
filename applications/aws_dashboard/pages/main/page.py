"""Main: The main SageWorks Web Interface to view, interact, and manage SageWorks Artifacts"""

from dash import register_page
import dash

# SageWorks Imports
from sageworks.views.artifacts_web_view import ArtifactsWebView
from sageworks.web_components import table

# Local Imports
from .layout import main_layout
from . import callbacks

register_page(
    __name__,
    path="/",
    name="SageWorks",
)

# Create a table for each AWS Service and Artifact Type
tables = dict()
tables["INCOMING_DATA"] = table.Table().create_component(
    "INCOMING_DATA",
    header_color="rgb(70, 70, 110)",
    max_height=135,
)
tables["GLUE_JOBS"] = table.Table().create_component(
    "GLUE_JOBS",
    header_color="rgb(70, 70, 110)",
    max_height=135,
)
tables["DATA_SOURCES"] = table.Table().create_component(
    "DATA_SOURCES",
    header_color="rgb(120, 70, 70)",
)
tables["FEATURE_SETS"] = table.Table().create_component(
    "FEATURE_SETS",
    header_color="rgb(110, 110, 70)",
)
tables["MODELS"] = table.Table().create_component(
    "MODELS",
    header_color="rgb(60, 100, 60)",
    max_height=235,
)
tables["ENDPOINTS"] = table.Table().create_component("ENDPOINTS", header_color="rgb(100, 60, 100)", max_height=235)

# Set up our components
components = {
    "incoming_data": tables["INCOMING_DATA"],
    "glue_jobs": tables["GLUE_JOBS"],
    "data_sources": tables["DATA_SOURCES"],
    "feature_sets": tables["FEATURE_SETS"],
    "models": tables["MODELS"],
    "endpoints": tables["ENDPOINTS"],
}

# Set up our layout (Dash looks for a var called layout)
layout = main_layout(**components)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
web_view = ArtifactsWebView()

# Setup our callbacks/connections
app = dash.get_app()
callbacks.refresh_data(app, web_view)
callbacks.update_artifact_tables(app)
