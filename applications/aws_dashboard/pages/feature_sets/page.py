"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import table, figures_plots, data_and_feature_details
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
data_details = data_and_feature_details.create("feature_set_details", details)

# Create a box plot of all the numeric columns in the sample rows
smart_sample_rows = feature_set_broker.feature_set_smart_sample(0)
violin = figures_plots.create("feature_set_violin_plot",
                                    smart_sample_rows, 
                                    plot_type="violin",
                                    figure_args={"box_visible": True, "meanline_visible": True, "showlegend": False, "points": "all"},
                                    max_plots=48)

# Create our components
components = {
    "feature_sets_table": feature_sets_table,
    "feature_set_sample_rows": feature_set_sample_rows,
    "feature_set_details": data_details,
    "violin_plot": violin,
}

# Set up our layout (Dash looks for a var called layout)
layout = feature_sets_layout(**components)

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
