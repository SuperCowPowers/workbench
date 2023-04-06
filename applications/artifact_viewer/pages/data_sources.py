"""HelloWorld: A SageWorks HelloWorld Application"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.data_sources_layout import data_sources_layout
import pages.callbacks.data_sources_callbacks as callbacks

register_page(__name__, path='/data_sources')


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
artifacts_summary = sageworks_artifacts.view_data()

# Grab the Artifact Information DataFrame for each AWS Service and pass it to the table creation
tables = dict()
tables["DATA_SOURCES_DETAILS"] = table.create(
    "DATA_SOURCES_DETAILS",
    artifacts_summary["DATA_SOURCES"],
    header_color="rgb(100, 60, 60)",
    markdown_columns=["Name"],
)

# Create our components
components = {
    "data_sources_details": tables["DATA_SOURCES_DETAILS"]
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app, sageworks_artifacts)
callbacks.update_data_source_table(app)

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(components)
