"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import histogram, box_plot
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.data_sources_layout import data_sources_layout
import pages.callbacks.data_sources_callbacks as callbacks

register_page(__name__, path="/data_sources")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
data_sources_summary = sageworks_artifacts.data_sources_summary()

# Create a table to display the data sources
data_sources_table = table.create(
    "DATA_SOURCES_DETAILS",
    data_sources_summary,
    header_color="rgb(100, 60, 60)",
    row_select="single",
    markdown_columns=["Name"],
)

# Create a fake scatter plot
histo = histogram.create()
box = box_plot.create()

# Create our components
components = {
    "data_sources_details": data_sources_table,
    "histo_plot": histo,
    "box_plot": box,
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_data_sources_table(app, sageworks_artifacts)

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(components)
