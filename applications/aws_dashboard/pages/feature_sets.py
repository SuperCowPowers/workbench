"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.web_components import scatter_plot
from sageworks.views.web_artifacts_summary import WebArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.feature_sets_layout import feature_sets_layout
import pages.callbacks.feature_sets_callbacks as callbacks

register_page(__name__, path="/feature_sets")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = WebArtifactsSummary()
feature_sets_summary = sageworks_artifacts.feature_sets_summary()

# Create a table to display the feature sets
feature_sets_table = table.create(
    "FEATURE_SETS_DETAILS",
    feature_sets_summary,
    header_color="rgb(100, 100, 60)",
    row_select="single",
    markdown_columns=["Feature Group"],
)

# Create a fake scatter plot
scatter1 = scatter_plot.create()
scatter2 = scatter_plot.create(variant=2)

# Create our components
components = {"feature_sets_details": feature_sets_table, "scatter1": scatter1, "scatter2": scatter2}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_feature_sets_table(app, sageworks_artifacts)

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(components)
