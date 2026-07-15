"""Contests: A Workbench Web Interface to view champion/challenger model contests"""

from dash import register_page

# Local Imports
from .layout import contests_layout
from . import callbacks

# Workbench Imports
from workbench.web_interface.page_views.contests_page_view import ContestsPageView

# Register this page with Dash
register_page(
    __name__,
    path="/contests",
    name="Workbench - Model Contests",
)

# Set up our layout (Dash looks for a var called layout)
layout = contests_layout()

# Grab a view that gives us the published contest reports
contests_view = ContestsPageView()

# Server-side data refresh + clientside render
callbacks.contest_data_refresh(contests_view)
callbacks.setup_render_callback()
