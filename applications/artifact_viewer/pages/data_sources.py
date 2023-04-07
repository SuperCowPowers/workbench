"""HelloWorld: A SageWorks HelloWorld Application"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.data_sources_layout import data_sources_layout
import pages.callbacks.data_sources_callbacks as callbacks

register_page(__name__, path="/data_sources")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
data_sources_summary = sageworks_artifacts.data_sources_summary()

# Create a table to display the data sources
data_sources_table = table.create(
    "DATA_SOURCES_DETAILS", data_sources_summary, header_color="rgb(100, 60, 60)", markdown_columns=["Name"]
)

# Create our components
components = {"data_sources_details": data_sources_table}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_data_sources_table(app, sageworks_artifacts)

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(components)
