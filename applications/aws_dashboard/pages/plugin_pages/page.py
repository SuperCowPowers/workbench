"""Plugin Page:  A SageWorks Plugin Web Interface"""
from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import plugin_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import table, plugin_loader
from sageworks.web_components.plugin_interface import PluginType
from sageworks.views.model_web_view import ModelWebView
from sageworks.utils.config_manager import ConfigManager

# Register this page with Dash
register_page(
    __name__,
    path="/plugin",
    name="SageWorks - Plugin",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Grab a view that gives us a summary of the Models in SageWorks
model_broker = ModelWebView()

# Create a table to display the models
models_table = table.Table().create_component(
    "plugin_table", header_color="rgb(60, 60, 60)", row_select="multi", max_height=400
)

# Capture our components in a dictionary to send off to the layout
components = {
    "plugin_table": models_table,
}

# Load the plugins from the sageworks_plugins directory
plugin_dir = ConfigManager().get_config("SAGEWORKS_PLUGINS")
plugins = plugin_loader.load_plugins_from_dir(plugin_dir, PluginType.CUSTOM)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = plugin_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_plugin_table(app)

# Callback for the plugin/model table
callbacks.table_row_select(app, "plugin_table")

# For each plugin, set up a callback to update the plugin figure
for plugin in plugins:
    callbacks.update_plugin(app, plugin, model_broker)
