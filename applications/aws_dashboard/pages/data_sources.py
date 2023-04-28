"""DataSources:  A SageWorks Web Interface to view, interact, and manage Data Sources"""
import pandas as pd
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import histogram, box_plot
from sageworks.views.data_source_view import DataSourceView
from sageworks.web_components import table

# Local Imports
from pages.layout.data_sources_layout import data_sources_layout
import pages.callbacks.data_sources_callbacks as callbacks

register_page(__name__, path="/data_sources")


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the DataSources in SageWorks
data_source_view = DataSourceView()
summary_data = data_source_view.data_sources_summary()

# Create a table to display the data sources
data_sources_summary = table.create(
    "data_sources_summary",
    summary_data,
    header_color="rgb(100, 60, 60)",
    row_select="single",
    markdown_columns=["Name"],
)

# Grab the first 5 rows of the first data source
if data_source_view.data_sources_meta:
    first_data_uuid = list(data_source_view.data_sources_meta.keys())[0]
    sample_rows = data_source_view.data_source_sample(first_data_uuid).head(5)
else:
    sample_rows = pd.DataFrame()
data_source_sample_rows = table.create(
    "data_source_sample_rows",
    sample_rows,
    header_color="rgb(60, 60, 100)",
    max_height="300px"
)

# Create a fake scatter plot
histo = histogram.create()
box = box_plot.create()

# Create our components
components = {
    "data_sources_summary": data_sources_summary,
    "data_source_sample_rows": data_source_sample_rows,
    "histo_plot": histo,
    "box_plot": box,
}

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)

# FIXME: Updating the table somehow breaks the row selection callback
# callbacks.update_data_sources_table(app, data_source_view)

# Callback for the data sources table
callbacks.table_row_select(app, "data_sources_summary")

# Set up our layout (Dash looks for a var called layout)
layout = data_sources_layout(components)
