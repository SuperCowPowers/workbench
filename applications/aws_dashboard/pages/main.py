"""Main: The main SageWorks Web Interface to view, interact, and manage SageWorks Artifacts"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.views.web_artifacts_summary import WebArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.main_layout import main_layout
import pages.callbacks.main_callbacks as callbacks

register_page(__name__, path="/")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
web_artifacts_summary = WebArtifactsSummary()
sageworks_artifacts = web_artifacts_summary.view_data()

# Grab the Artifact Information DataFrame for each AWS Service and pass it to the table creation
tables = dict()
tables["INCOMING_DATA"] = table.create(
    "INCOMING_DATA",
    sageworks_artifacts["INCOMING_DATA"],
    header_color="rgb(60, 60, 100)",
)
tables["DATA_SOURCES"] = table.create(
    "DATA_SOURCES",
    sageworks_artifacts["DATA_SOURCES"],
    header_color="rgb(100, 60, 60)",
    markdown_columns=["Name"],
)
tables["FEATURE_SETS"] = table.create(
    "FEATURE_SETS",
    sageworks_artifacts["FEATURE_SETS"],
    header_color="rgb(100, 100, 60)",
    markdown_columns=["Feature Group"],
)
tables["MODELS"] = table.create(
    "MODELS",
    sageworks_artifacts["MODELS"],
    header_color="rgb(60, 100, 60)",
    markdown_columns=["Model Group"],
)
tables["ENDPOINTS"] = table.create(
    "ENDPOINTS", sageworks_artifacts["ENDPOINTS"], header_color="rgb(100, 60, 100)", markdown_columns=["Name"]
)

# Create our components
components = {
    "incoming_data": tables["INCOMING_DATA"],
    "data_sources": tables["DATA_SOURCES"],
    "feature_sets": tables["FEATURE_SETS"],
    "models": tables["MODELS"],
    "endpoints": tables["ENDPOINTS"],
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app, web_artifacts_summary)
callbacks.update_artifact_tables(app)

# Set up our layout (Dash looks for a var called layout)
layout = main_layout(components)
