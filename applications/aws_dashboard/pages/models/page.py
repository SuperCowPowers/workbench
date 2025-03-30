"""Models:  A Workbench Web Interface to view, and interact with Models"""

from dash import register_page

# Local Imports
from .layout import models_layout
from . import callbacks

# Workbench Imports
from workbench.web_interface.components import model_plot
from workbench.web_interface.components.plugins import model_details, ag_table
from workbench.web_interface.components.plugin_interface import PluginPage
from workbench.utils.plugin_manager import PluginManager
from workbench.web_interface.page_views.models_page_view import ModelsPageView

# Register this page with Dash
register_page(
    __name__,
    path="/models",
    name="Workbench - Models",
)

# Create a table to display the models
models_table = ag_table.AGTable()
models_table_component = models_table.create_component("models_table", header_color="rgb(60, 100, 60)", max_height=270)

# Create a Markdown component to display the model details
my_model_details = model_details.ModelDetails()
model_details_component = my_model_details.create_component("model_details")

# Create a Model Plot component to display the model metrics
model_plot_component = model_plot.ModelPlot().create_component("model_plot")

# Capture our components in a dictionary to send off to the layout
components = {
    "models_table": models_table_component,
    "model_details": model_details_component,
    "model_plot": model_plot_component,
}

# Load any web components plugins of type 'model'
pm = PluginManager()
plugins = pm.get_list_of_web_plugins(plugin_page=PluginPage.MODEL)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.generate_component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(**components)

# Grab a view that gives us a summary of the Models in Workbench
model_view = ModelsPageView()

# Callback for anything we want to happen on page load
callbacks.on_page_load()

# Setup our callbacks/connections
callbacks.model_table_refresh(model_view, models_table)

# Callback for the model table
callbacks.update_model_plot_component()

# Our model details is a plugin, so we need to add it to the list
plugins.append(my_model_details)

# Set up callbacks for all the plugins
if plugins:
    callbacks.setup_plugin_callbacks(plugins)
