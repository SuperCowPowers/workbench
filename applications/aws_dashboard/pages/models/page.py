"""Models:  A Workbench Web Interface to view, and interact with Models"""

from dash import register_page, html

# Local Imports
from .layout import models_layout
from . import callbacks

# Workbench Imports
from workbench.web_interface.components.plugins import (
    model_details,
    ag_table,
    shap_summary_plot,
    scatter_plot,
    confusion_matrix,
)
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

# Create ScatterPlot plugin for regression models
my_scatter_plot = scatter_plot.ScatterPlot()
scatter_plot_component = my_scatter_plot.create_component("model_scatter_plot")

# Create ConfusionMatrix plugin for classification models
my_confusion_matrix = confusion_matrix.ConfusionMatrix()
confusion_matrix_component = my_confusion_matrix.create_component("model_confusion_matrix")

# Wrap both in visibility-controlled containers
# Show scatter plot by default (with "Waiting for Data..." placeholder)
# The callback will toggle visibility based on model type once data loads
model_plot_container = html.Div(
    [
        html.Div(scatter_plot_component, id="scatter-plot-container", style={"display": "block"}),
        html.Div(confusion_matrix_component, id="confusion-matrix-container", style={"display": "none"}),
    ],
    id="model-plot-container",
)

# Shap summary plot component
my_shap_plot = shap_summary_plot.ShapSummaryPlot()
shap_plot_component = my_shap_plot.create_component("shap_plot")


# Capture our components in a dictionary to send off to the layout
components = {
    "models_table": models_table_component,
    "model_details": model_details_component,
    "model_plot": model_plot_container,
    "shap_plot": shap_plot_component,
}

# Set up our layout (Dash looks for a var called layout)
layout = models_layout(**components)

# Grab a view that gives us a summary of the Models in Workbench
model_view = ModelsPageView()

# Callback for anything we want to happen on page load
callbacks.on_page_load()

# Setup our callbacks/connections
callbacks.model_table_refresh(model_view, models_table)

# All model visualization plugins
plugins = [my_model_details, my_shap_plot, my_scatter_plot, my_confusion_matrix]

# Set up callbacks for all the plugins (includes scatter plot and confusion matrix)
callbacks.setup_plugin_callbacks(plugins, my_scatter_plot, my_confusion_matrix)
