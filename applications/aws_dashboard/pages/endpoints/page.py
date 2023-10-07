"""Endpoints:  A SageWorks Web Interface to view and interact with Endpoints"""
import os
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import endpoints_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import table, model_markdown, plugin_loader
from sageworks.web_components.plugin_interface import PluginType
from sageworks.views.endpoint_web_view import EndpointWebView

# Register this page with Dash
register_page(
    __name__,
    path="/endpoints",
    name="SageWorks - Endpoints",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the Models in SageWorks
endpoint_broker = EndpointWebView()

# Create a table to display the endpoints
endpoints_table = table.Table().create_component(
    "endpoints_table",
    header_color="rgb(100, 60, 100)",
    row_select="single",
)

# Create a Markdown component to display the endpoint details
endpoint_details = model_markdown.ModelMarkdown().create_component("endpoint_details")

# Capture our components in a dictionary to send off to the layout
components = {
    "endpoints_table": endpoints_table,
    "endpoint_details": endpoint_details,
}

# Load the plugins from the sageworks_plugins directory
plugins = plugin_loader.load_plugins_from_dir(os.getenv("SAGEWORKS_PLUGINS"), PluginType.ENDPOINT)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = endpoints_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_endpoints_table(app)

# Callback for the endpoints table
callbacks.table_row_select(app, "endpoints_table")
callbacks.update_endpoint_details(app, endpoint_broker)

# For each plugin, set up a callback to update the plugin figure
for plugin in plugins:
    callbacks.update_plugin(app, plugin, endpoint_broker)
