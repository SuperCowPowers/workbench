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


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
web_view = ArtifactsWebView()
sageworks_artifacts = web_view.view_data()

# Add a 'delete' column for all the artifacts listings except for the INCOMING_DATA and GLUE_JOBS
for category in sageworks_artifacts:
    if category not in ["INCOMING_DATA", "GLUE_JOBS"]:
        sageworks_artifacts[category]["del"] = "<img src='../assets/trash.png' id='trash-icon'>"

# Grab the Artifact Information DataFrame for each AWS Service and pass it to the table creation
tables = dict()
tables["INCOMING_DATA"] = table.create(
    table_id="INCOMING_DATA",
    df=sageworks_artifacts["INCOMING_DATA"],
    header_color="rgb(60, 60, 100)",
    markdown_columns=["Name"],
)
tables["GLUE_JOBS"] = table.create(
    table_id="GLUE_JOBS",
    df=sageworks_artifacts["GLUE_JOBS"],
    header_color="rgb(60, 60, 100)",
    markdown_columns=["Name"],
)
tables["DATA_SOURCES"] = table.create(
    table_id="DATA_SOURCES",
    df=sageworks_artifacts["DATA_SOURCES"],
    header_color="rgb(100, 60, 60)",
    markdown_columns=["Name", "del"],
)
tables["FEATURE_SETS"] = table.create(
    table_id="FEATURE_SETS",
    df=sageworks_artifacts["FEATURE_SETS"],
    header_color="rgb(100, 100, 60)",
    markdown_columns=["Feature Group", "del"],
)
tables["MODELS"] = table.create(
    table_id="MODELS",
    df=sageworks_artifacts["MODELS"],
    header_color="rgb(60, 100, 60)",
    markdown_columns=["Model Group", "del"],
)
tables["ENDPOINTS"] = table.create(
    table_id="ENDPOINTS",
    df=sageworks_artifacts["ENDPOINTS"],
    header_color="rgb(100, 60, 100)",
    markdown_columns=["Name", "del"],
)

# Create our components
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

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app, web_view)
callbacks.update_artifact_tables(app)
callbacks.delete_artifact_callbacks(app, web_view)
