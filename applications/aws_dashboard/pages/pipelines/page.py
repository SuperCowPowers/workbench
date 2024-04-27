"""Pipelines:  A SageWorks Web Interface to view and interact with Pipelines"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import pipelines_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components.plugins import ag_table, pipeline_details
from sageworks.web_components.plugin_interface import PluginPage
from sageworks.utils.plugin_manager import PluginManager

# Register this page with Dash
register_page(
    __name__,
    path="/pipelines",
    name="SageWorks - Pipelines",
)

# Put the components into 'dark' mode
load_figure_template("darkly")

# Create a table to display the pipelines
pipeline_table = ag_table.AGTable()
table_component = pipeline_table.create_component("pipelines_table")

# Create a Markdown component to display the pipeline details
pipeline_details = pipeline_details.PipelineDetails()
details_component = pipeline_details.create_component("pipeline_details")

# Capture our components in a dictionary to send off to the layout
components = {"pipelines_table": table_component, "pipeline_details": details_component}

# Load any web components plugins of type 'pipeline'
pm = PluginManager()
plugins = pm.get_list_of_web_plugins(plugin_page=PluginPage.PIPELINE)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.generate_component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = pipelines_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_pipelines_table(pipeline_table)

# We're going to add the details component to the plugins list
plugins.append(pipeline_details)

# For all the plugins we have we'll call their update_properties method
if plugins:
    callbacks.setup_plugin_callbacks(plugins)
