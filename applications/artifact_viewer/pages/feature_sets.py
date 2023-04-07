"""HelloWorld: A SageWorks HelloWorld Application"""
from dash import register_page
import dash

# SageWorks Imports
from sageworks.views.artifacts_summary import ArtifactsSummary
from sageworks.web_components import table

# Local Imports
from pages.layout.feature_sets_layout import feature_sets_layout
import pages.callbacks.feature_sets_callbacks as callbacks

register_page(__name__, path='/feature_sets')


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Grab a view that gives us a summary of all the artifacts currently in SageWorks
sageworks_artifacts = ArtifactsSummary()
feature_sets_summary = sageworks_artifacts.feature_sets_summary()

# Grab the Artifact Information DataFrame for each AWS Service and pass it to the table creation
tables = dict()
tables["FEATURE_SETS_DETAILS"] = table.create(
    "FEATURE_SETS_DETAILS",
    feature_sets_summary,
    header_color="rgb(100, 100, 60)",
    markdown_columns=["Feature Group"],
)

# Create our components
components = {
    "feature_sets_details": tables["FEATURE_SETS_DETAILS"]
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_feature_sets_table(app, sageworks_artifacts)

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(components)
