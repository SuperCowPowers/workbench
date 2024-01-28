"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import table, data_details_markdown, violin_plots, correlation_matrix
from sageworks.views.feature_set_web_view import FeatureSetWebView

# Local Imports
from .layout import feature_sets_layout
from . import callbacks

register_page(__name__, path="/feature_sets", name="SageWorks - Feature Sets")

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the FeatureSets in SageWorks
feature_set_broker = FeatureSetWebView()

# Create a table to display the feature sets
feature_sets_table = table.Table().create_component(
    "feature_sets_table",
    header_color="rgb(110, 110, 70)",
    row_select="single",
)

# Grab smart_sample rows from the first Feature Set
feature_set_sample_rows = table.Table().create_component(
    "feature_set_sample_rows",
    header_color="rgb(70, 70, 110)",
    max_height=250,
)

# Feature Set Details
data_details = data_details_markdown.DataDetailsMarkdown().create_component("feature_set_details")

# Create a violin plot of all the numeric columns in the Feature Set
violin = violin_plots.ViolinPlots().create_component("feature_set_violin_plot")

# Create a correlation matrix component
corr_matrix = correlation_matrix.CorrelationMatrix().create_component("feature_set_correlation_matrix")


# Create our components
components = {
    "feature_sets_table": feature_sets_table,
    "feature_set_sample_rows": feature_set_sample_rows,
    "feature_set_details": data_details,
    "violin_plot": violin,
    "correlation_matrix": corr_matrix,
}

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()

# Periodic update to the data sources summary table
callbacks.update_feature_sets_table(app)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "feature_sets_table")
callbacks.update_feature_set_details(app, feature_set_broker)
callbacks.update_feature_set_sample_rows(app, feature_set_broker)

# Callbacks for selections
callbacks.violin_plot_selection(app)
callbacks.reorder_sample_rows(app)
callbacks.correlation_matrix_selection(app)
