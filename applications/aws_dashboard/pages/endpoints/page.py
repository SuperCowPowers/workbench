"""Endpoints:  A SageWorks Web Interface to view and interact with Endpoints"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import endpoints_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import table, endpoint_details, endpoint_metric_plots
from sageworks.web_components.plugin_interface import PluginPage
from sageworks.views.endpoint_web_view import EndpointWebView
from sageworks.utils.plugin_manager import PluginManager

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
    "endpoints_table", header_color="rgb(100, 60, 100)", row_select="single", max_height=270
)

# Create a Markdown component to display the endpoint details
endpoint_details = endpoint_details.EndpointDetails()
endpoint_details_component = endpoint_details.create_component("endpoint_details")

# Create a component to display the endpoint metrics
endpoint_metrics = endpoint_metric_plots.EndpointMetricPlots().create_component("endpoint_metrics")

# Capture our components in a dictionary to send off to the layout
components = {
    "endpoints_table": endpoints_table,
    "endpoint_details": endpoint_details_component,
    "endpoint_metrics": endpoint_metrics,
}

# Load any web components plugins of type 'endpoint'
pm = PluginManager()
plugins = pm.get_list_of_web_plugins(plugin_page=PluginPage.ENDPOINT)

# Add the plugins to the components dictionary
for plugin in plugins:
    component_id = plugin.generate_component_id()
    components[component_id] = plugin.create_component(component_id)

# Set up our layout (Dash looks for a var called layout)
layout = endpoints_layout(**components)

# Setup our callbacks/connections
app = dash.get_app()
callbacks.update_endpoints_table(app)
endpoint_details.register_callbacks("endpoints_table")

# Callback for the endpoints table
callbacks.table_row_select(app, "endpoints_table")
callbacks.update_endpoint_metrics(app, endpoint_broker)

# For all the plugins we have we'll call their update_properties method
if plugins:
    callbacks.setup_plugin_callbacks(plugins)
