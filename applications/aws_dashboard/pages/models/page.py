"""Models:  A SageWorks Web Interface to view, and interact with Models"""

from dash import register_page
import dash
from dash_bootstrap_templates import load_figure_template

# Local Imports
from .layout import models_layout
from . import callbacks

# SageWorks Imports
from sageworks.web_components import (
    table,
    model_details_markdown,
    model_metrics_markdown,
    inference_run_selector,
    model_plot,
)
from sageworks.web_components.plugin_interface import PluginType
from sageworks.views.model_web_view import ModelWebView
from sageworks.utils.plugin_manager import PluginManager

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
    "models_table", header_color="rgb(60, 100, 60)", row_select="single", max_height=270
)

# Create a Markdown component to display the model details
model_details = model_details_markdown.ModelDetailsMarkdown().create_component("model_details")

# Create a Inference Run Selector component
inf_run_sel_component = inference_run_selector.InferenceRunSelector().create_component("inference_run_selector")

# Create a Markdown component to display model metrics
model_metrics = model_metrics_markdown.ModelMetricsMarkdown().create_component("model_metrics")

# Create a Model Plot component to display the model metrics
model_plot_component = model_plot.ModelPlot().create_component("model_plot")

# Capture our components in a dictionary to send off to the layout
components = {
    "models_table": models_table,
    "inference_run_selector": inf_run_sel_component,
    "model_details": model_details,
    "model_metrics": model_metrics,
    "model_plot": model_plot_component,
}

# Load any web components plugins of type 'model'
pm = PluginManager()
plugins = pm.get_list_of_web_plugins(web_plugin_type=PluginType.MODEL)

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
callbacks.update_inference_run_selector(app)
callbacks.update_model_detail_component(app)
callbacks.update_model_metrics_component(app)
callbacks.update_model_plot_component(app)

# For each plugin, set up a callback to update the plugin figure
for plugin in plugins:
    callbacks.update_plugin(app, plugin, model_broker)
