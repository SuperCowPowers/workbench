"""Status:  A Workbench Web Interface to view Status Details about the Workbench Dashboard"""

from dash import register_page
from dash import html

# Workbench Imports
from workbench.web_interface.components.plugins import dashboard_status
from workbench.utils.config_manager import ConfigManager


# Register this page with Dash
register_page(
    __name__,
    path="/status",
    name="Workbench - Dashboard Status",
)

# Grab the Workbench ConfigManager
cm = ConfigManager()
config_details = cm.get_all_config()

# Create a Markdown component to display the license details
markdown_details = dashboard_status.DashboardStatus()
details_component = markdown_details.create_component("status_details")
updated_properties = markdown_details.update_properties(config_details)

# Set the properties directly on the server side before initializing the layout
for (component_id, prop), value in zip(markdown_details.properties, updated_properties):
    for child in details_component.children:
        if child.id == component_id:
            setattr(child, prop, value)
            break

# Simple layout for the license details
layout = html.Div(
    children=[
        html.H2("Workbench Dashboard Status"),
        details_component,
    ],
    style={"padding": "12px 30px 30px 30px"},
)
