"""Pipelines:  A Workbench Web Interface to view and interact with Pipelines"""

from dash import register_page

# Local Imports
from .layout import pipelines_layout
from . import callbacks

# Workbench Imports
from workbench.web_interface.components.plugins import pipeline_details, ag_table
from workbench.web_interface.components.plugin_interface import PluginPage
from workbench.web_interface.page_views.pipelines_page_view import PipelinesPageView
from workbench.utils.plugin_manager import PluginManager

# Register this page with Dash
register_page(
    __name__,
    path="/pipelines",
    name="Workbench - Pipelines",
)

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

# Grab a view that gives us a summary of the Pipelines in Workbench
pipelines_view = PipelinesPageView()

# Callback for anything we want to happen on page load
callbacks.on_page_load()

# Setup our callbacks/connections
callbacks.pipeline_table_refresh(pipelines_view, pipeline_table)

# We're going to add the details component to the plugins list
plugins.append(pipeline_details)

# For all the plugins we have we'll call their update_properties method
if plugins:
    callbacks.setup_plugin_callbacks(plugins)
