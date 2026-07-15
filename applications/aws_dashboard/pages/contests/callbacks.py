"""Callbacks for the Model Contests page

Follows the custom-rendered page contract (see pages/ml_pipelines/callbacks.py):
a server-side @callback fills a dcc.Store from the PageView, and a clientside
render callback draws it. Python fills data, JS owns pixels.
"""

import logging
from dash import callback, clientside_callback, Input, Output, ClientsideFunction

# Workbench Imports
from workbench.web_interface.page_views.contests_page_view import ContestsPageView

# Get the Workbench logger
log = logging.getLogger("workbench")


def contest_data_refresh(page_view: ContestsPageView):
    """Server-side: refresh the contest reports and push them to the Store"""

    @callback(
        Output("contests_data", "data"),
        Input("contests_refresh", "n_intervals"),
    )
    def _refresh(_n):
        page_view.refresh()
        return page_view.contests()


def setup_render_callback():
    """Clientside: render the Store data into #contests-root (card grid + expand)"""
    clientside_callback(
        ClientsideFunction(namespace="contests", function_name="render"),
        Output("contests_render_signal", "children"),
        Input("contests_data", "data"),
    )
