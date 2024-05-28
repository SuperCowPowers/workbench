"""Models:  A SageWorks Web Interface to view, and interact with Models"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import models_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import table, model_plot
from sageworks.web_components.plugins import model_details
from sageworks.web_components.plugin_interface import PluginPage
from sageworks.utils.plugin_manager import PluginManager

# Register this page with Dash
register_page(
    __name__,
    path="/models",
    name="SageWorks - Models",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Create a table to display the models
models_table = table.Table().create_component(
    "models_table", header_color="rgb(60, 100, 60)", row_select="single", max_height=270
)

# Create a Markdown component to display the model details
model_details = model_details.ModelDetails()
model_details_component = model_details.create_component("model_details")

# Create a Model Plot component to display the model metrics
model_plot_component = model_plot.ModelPlot().create_component("model_plot")

# Capture our components in a dictionary to send off to the layout
components = {
    "models_table": models_table,
    "model_details": model_details_component,
    "model_plot": model_plot_component,
}

# Load any web components plugins of type 'model'
pm = PluginManager()
plugins = pm.get_list_of_web_plugins(plugin_page=PluginPage.MODEL)

# Our model details is a plugin, so we need to add it to the list
plugins.append(model_details)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.generate_component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_models_table(app)

# Callback for the model table
callbacks.table_row_select(app, "models_table")
callbacks.update_model_plot_component(app)

# Set up callbacks for all the plugins
if plugins:
    callbacks.setup_plugin_callbacks(plugins)
