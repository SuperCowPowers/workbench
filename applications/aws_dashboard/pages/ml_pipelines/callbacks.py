"""Callbacks for the ML Pipelines Subpage Web User Interface

Custom-rendered page contract (reuse this for any JS-rendered Dashboard page):

  1. Store    - a dcc.Store that a server-side @callback fills from a PageView.
  2. Render   - a clientside_callback(ClientsideFunction(namespace, "render"))
                triggered by the Store; the JS owns one root Div and draws into it.
  3. Assets   - the render JS/CSS live in assets/<page>/ (Dash serves them by
                namespace, not filename; keep the namespace == the page name).

Python fills data, JS owns pixels. No server round-trips for interaction.
"""

import logging
from dash import callback, clientside_callback, Input, Output, ClientsideFunction

# Workbench Imports
from workbench.web_interface.page_views.ml_pipelines_page_view import MLPipelinesPageView

# Get the Workbench logger
log = logging.getLogger("workbench")


def pipeline_data_refresh(page_view: MLPipelinesPageView):
    """Server-side: refresh the pipeline hierarchy and push it to the Store"""

    @callback(
        Output("ml_pipelines_data", "data"),
        Input("ml_pipelines_refresh", "n_intervals"),
    )
    def _refresh(_n):
        page_view.refresh()
        return page_view.pipelines()


def setup_render_callback():
    """Clientside: render the Store data into #ml-pipelines-root (cards + graph view)"""
    clientside_callback(
        ClientsideFunction(namespace="ml_pipelines", function_name="render"),
        Output("ml_pipelines_render_signal", "children"),
        Input("ml_pipelines_data", "data"),
    )
