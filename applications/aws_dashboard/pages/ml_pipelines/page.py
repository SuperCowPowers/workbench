"""ML Pipelines:  A Workbench Web Interface to view ML Pipelines"""

from dash import register_page

# Local Imports
from .layout import ml_pipelines_layout
from . import callbacks

# Workbench Imports
from workbench.web_interface.page_views.ml_pipelines_page_view import MLPipelinesPageView

# Register this page with Dash
register_page(
    __name__,
    path="/ml_pipelines",
    name="Workbench - ML Pipelines",
)

# Set up our layout (Dash looks for a var called layout)
layout = ml_pipelines_layout()

# Grab a view that gives us the ML Pipeline hierarchy
pipelines_view = MLPipelinesPageView()

# Server-side data refresh + clientside render
callbacks.pipeline_data_refresh(pipelines_view)
callbacks.setup_render_callback()
