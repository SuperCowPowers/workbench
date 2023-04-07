"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
import os
from dash import register_page
import dash

# SageWorks Imports
from sageworks.web_components import model_data
from sageworks.web_components import scatter_plot
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.feature_sets_layout import feature_sets_layout
import pages.callbacks.feature_sets_callbacks as callbacks

register_page(__name__, path="/feature_sets")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Read in our fake model data
data_path = os.path.join(os.path.dirname(__file__), "data/toy_data.csv")
fake_model_info = model_data.ModelData(data_path)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
feature_sets_summary = sageworks_artifacts.feature_sets_summary()

# Create a table to display the feature sets
feature_sets_table = table.create(
    "FEATURE_SETS_DETAILS",
    feature_sets_summary,
    header_color="rgb(100, 100, 60)",
    row_select="single",
    markdown_columns=["Feature Group"]
)

# Create a fake scatter plot
model_df = fake_model_info.get_model_df()
scatter = scatter_plot.create(model_df)

# Create our components
components = {
    "feature_sets_details": feature_sets_table,
    "scatter_plot": scatter
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_feature_sets_table(app, sageworks_artifacts)

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(components)
