"""Models:  A SageWorks Web Interface to view, and interact with Models"""
import os
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import models_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import table, model_markdown, plugin_loader
from sageworks.web_components.plugin_interface import PluginType
from sageworks.views.model_web_view import ModelWebView

# Register this page with Dash
register_page(
    __name__,
    path="/models",
    name="SageWorks - Models",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the Models in SageWorks
model_broker = ModelWebView()

# Create a table to display the models
models_table = table.Table().create_component(
    "models_table",
    header_color="rgb(60, 100, 60)",
    row_select="single",
)

# Create a Markdown component to display the model details
model_details = model_markdown.ModelMarkdown().create_component("model_details")

# Capture our components in a dictionary to send off to the layout
components = {
    "models_table": models_table,
    "model_details": model_details,
}

# Load the plugins from the sageworks_plugins directory
plugins = plugin_loader.load_plugins_from_dir(os.getenv("SAGEWORKS_PLUGINS"), PluginType.MODEL)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_models_table(app)

# Callback for the model table
callbacks.table_row_select(app, "models_table")
callbacks.update_model_details(app, model_broker)

# For each plugin, set up a callback to update the plugin figure
for plugin in plugins:
    callbacks.update_plugin(app, plugin, model_broker)
