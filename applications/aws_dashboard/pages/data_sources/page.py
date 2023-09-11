"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import table, data_details_markdown, distribution_plots, heatmap
from sageworks.views.data_source_web_view import DataSourceWebView
from sageworks.utils.pandas_utils import corr_df_from_artifact_info

# Local Imports
from .layout import data_sources_layout
from . import callbacks

register_page(
    __name__,
    path="/data_sources",
    name="SageWorks - Data Sources",
)

# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the DataSources in SageWorks
data_source_broker = DataSourceWebView()
data_source_rows = data_source_broker.data_sources_summary()

# Create a table to display the data sources
data_sources_table = table.create(
    "data_sources_table",
    data_source_rows,
    header_color="rgb(100, 60, 60)",
    row_select="single",
    markdown_columns=["Name"],
)

# Grab sample rows from the first data source
smart_sample_rows = data_source_broker.data_source_smart_sample(0)
data_source_sample_rows = table.create(
    "data_source_sample_rows",
    smart_sample_rows,
    header_color="rgb(60, 60, 100)",
    max_height="300px",
    color_column="outlier_group",
)

# Data Source Details
details = data_source_broker.data_source_details(0)
data_details = data_details_markdown.create("data_source_details", details)

# Create a violin plot of all the numeric columns in the Data Source
violin = distribution_plots.create(
    "data_source_violin_plot",
    smart_sample_rows,
    plot_type="violin",
    figure_args={
        "box_visible": True,
        "meanline_visible": True,
        "showlegend": False,
        "points": "all",
        "spanmode": "hard",
    },
    max_plots=48,
)

# Create a correlation matrix of all the numeric columns in the Data Source
corr_df = corr_df_from_artifact_info(details)
corr_matrix = heatmap.create("correlation_matrix", corr_df)

# Create our components
components = {
    "data_sources_table": data_sources_table,
    "data_source_details": data_details,
    "data_source_sample_rows": data_source_sample_rows,
    "violin_plot": violin,
    "correlation_matrix": corr_matrix,
}

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()

# Refresh our data timer
callbacks.refresh_data_timer(app)

# Periodic update to the data sources summary table
callbacks.update_data_sources_table(app, data_source_broker)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "data_sources_table")
callbacks.update_data_source_details(app, data_source_broker)
callbacks.update_data_source_sample_rows(app, data_source_broker)
callbacks.update_violin_plots(app, data_source_broker)
callbacks.update_correlation_matrix(app, data_source_broker)

# Callbacks for selections
callbacks.violin_plot_selection(app)
callbacks.reorder_sample_rows(app)
callbacks.correlation_matrix_selection(app)
