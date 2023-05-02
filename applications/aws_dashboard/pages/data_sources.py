"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import violin_plot
from sageworks.views.web_data_source_view import WebDataSourceView
from sageworks.web_components import table

# Local Imports
from pages.layout.data_sources_layout import data_sources_layout
import pages.callbacks.data_sources_callbacks as callbacks

register_page(__name__, path="/data_sources")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the DataSources in SageWorks
web_data_source_view = WebDataSourceView()
summary_data = web_data_source_view.data_sources_summary()

# Create a table to display the data sources
data_sources_summary = table.create(
    "data_sources_summary",
    summary_data,
    header_color="rgb(100, 60, 60)",
    row_select="single",
    markdown_columns=["Name"],
)

# Grab the a sample of rows from the first data source
sample_rows = web_data_source_view.data_source_sample(0)
data_source_sample_rows = table.create(
    "data_source_sample_rows",
    sample_rows,
    header_color="rgb(60, 60, 100)",
    max_height="200px",
    fixed_headers=True
)

# Create a box plot of all the numeric columns in the sample rows
# histo = histogram.create()
violin = violin_plot.create(sample_rows)

# Create our components
components = {
    "data_sources_summary": data_sources_summary,
    "data_source_sample_rows": data_source_sample_rows,
    "violin_plot": violin
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)

# Periodic update to the data sources summary
# FIXME: This causes issues with the table selections
# callbacks.update_data_sources_summary(app, web_data_source_view)

# Callbacks for when a data source is selected
callbacks.table_row_select(app, "data_sources_summary")
callbacks.update_sample_rows_header(app)
callbacks.update_data_source_sample_rows(app, web_data_source_view)
callbacks.update_violin_plots(app, web_data_source_view)

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(components)
