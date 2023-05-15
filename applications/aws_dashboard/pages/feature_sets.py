"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import violin_plot, table, feature_set_details
from sageworks.views.feature_set_web_view import FeatureSetWebView

# Local Imports
from pages.layout.feature_sets_layout import feature_sets_layout
import pages.callbacks.feature_sets_callbacks as callbacks

register_page(__name__, path="/feature_sets", name="Feature Sets")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the DataSources in SageWorks
feature_set_broker = FeatureSetWebView()
feature_set_rows = feature_set_broker.feature_sets_summary()

# Create a table to display the data sources
feature_sets_table = table.create(
    "feature_sets_table",
    feature_set_rows,
    header_color="rgb(100, 100, 60)",
    row_select="single",
    markdown_columns=["Feature Group"],
)

# Grab sample rows from the first data source
sample_rows = feature_set_broker.feature_set_sample(0)
feature_set_sample_rows = table.create(
    "feature_set_sample_rows",
    sample_rows,
    header_color="rgb(60, 60, 100)",
    max_height="200px",
)

# Data Source Details
details = feature_set_broker.feature_set_details(0)
data_details = feature_set_details.create("feature_set_details", details)

# Create a box plot of all the numeric columns in the sample rows
smart_sample_rows = feature_set_broker.feature_set_smart_sample(0)
violin = violin_plot.create("feature_set_violin_plot", smart_sample_rows)

# Create our components
components = {
    "feature_sets_table": feature_sets_table,
    "feature_set_details": data_details,
    "feature_set_sample_rows": feature_set_sample_rows,
    "violin_plot": violin,
}

# Setup our callbacks/connections
app = dash.get_app()

# Refresh our data timer
callbacks.refresh_data_timer(app)

# Periodic update to the data sources summary table
callbacks.update_feature_sets_table(app, feature_set_broker)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "feature_sets_table")
callbacks.update_feature_set_details(app, feature_set_broker)
callbacks.update_feature_set_sample_rows(app, feature_set_broker)
callbacks.update_violin_plots(app, feature_set_broker)

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(components)
