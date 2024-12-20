"""License:  A Workbench Web Interface to view License Details"""

from dash import register_page
from dash import html

# Workbench Imports
from workbench.web_interface.components.plugins import license_details
from workbench.utils.config_manager import ConfigManager
from workbench.utils.license_manager import LicenseManager

# Register this page with Dash
register_page(
    __name__,
    path="/license",
    name="Workbench - License",
)

# Grab the API Key from the Workbench ConfigManager
cm = ConfigManager()
api_key = cm.get_config("WORKBENCH_API_KEY")
license_api_key = cm.get_config("LICENSE_API_KEY")
my_license_info = LicenseManager.load_api_license(aws_account_id=None, api_key=api_key, license_api_key=license_api_key)

# Create a Markdown component to display the license details
markdown_details = license_details.LicenseDetails()
details_component = markdown_details.create_component("license_details")
updated_properties = markdown_details.update_properties(my_license_info)

# Set the properties directly on the server side before initializing the layout
for (component_id, prop), value in zip(markdown_details.properties, updated_properties):
    for child in details_component.children:
        if child.id == component_id:
            setattr(child, prop, value)
            break

# Simple layout for the license details
layout = html.Div(
    children=[
        html.H2("Workbench: License"),
        details_component,
    ],
    style={"padding": "12px 30px 30px 30px"},
)
