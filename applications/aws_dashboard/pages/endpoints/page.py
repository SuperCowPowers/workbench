"""Endpoints:  A SageWorks Web Interface to view, interact, and manage Endpoints"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components import line_chart
from sageworks.web_components import table

# Local Imports
from .layout import endpoints_layout
from . import callbacks

register_page(
    __name__,
    path="/endpoints",
    name="SageWorks - Endpoints",
)


# Okay feels a bit weird but Dash pages just have a bunch of top level code (no classes/methods)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Create a table to display the endpoints
endpoints_table = table.Table().create_component(
    "endpoints_table",
    header_color="rgb(100, 60, 100)",
    row_select="single",
)

# Create a fake scatter plot
endpoint_traffic = line_chart.create()

# Create our components
components = {
    "endpoints_table": endpoints_table,
    "endpoint_traffic": endpoint_traffic,
}

# Set up our layout (Dash looks for a var called layout)
layout = endpoints_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_last_updated(app)
callbacks.update_endpoints_table(app)
