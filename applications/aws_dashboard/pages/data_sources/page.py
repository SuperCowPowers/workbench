"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import table, data_details_markdown, violin_plots, correlation_matrix
from sageworks.views.data_source_web_view import DataSourceWebView

# Local Imports
from .layout import data_sources_layout
from . import callbacks

register_page(
    __name__,
    path="/data_sources",
    name="SageWorks - Data Sources",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the DataSources in SageWorks
data_source_broker = DataSourceWebView()
data_source_rows = data_source_broker.data_sources_summary()

# Create a table to display the data sources
data_sources_table = table.Table().create_component(
    "data_sources_table",
    header_color="rgb(120, 70, 70)",
    row_select="single",
)

# Create a table that sample rows from the currently selected  data source
data_source_sample_rows = table.Table().create_component(
    "data_source_sample_rows",
    header_color="rgb(70, 70, 110)",
    max_height=250,
)

# Data Source Details Markdown PANEL
data_details = data_details_markdown.DataDetailsMarkdown().create_component("data_source_details")

# Create a violin plot of all the numeric columns in the Data Source
violin = violin_plots.ViolinPlots().create_component("data_source_violin_plot")

# Create a correlation matrix of all the numeric columns in the Data Source
corr_matrix = correlation_matrix.CorrelationMatrix().create_component("data_source_correlation_matrix")

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

# Periodic update to the data sources summary table
callbacks.update_data_sources_table(app)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "data_sources_table")
callbacks.update_data_source_details(app, data_source_broker)
callbacks.update_data_source_sample_rows(app, data_source_broker)

# Callbacks for selections
callbacks.violin_plot_selection(app)
callbacks.reorder_sample_rows(app)
callbacks.correlation_matrix_selection(app)
