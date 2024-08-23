"""License:  A SageWorks Web Interface to view License Details"""

from dash import register_page
from dash import html
from dash_bootstrap_templates import load_figure_template

# SageWorks Imports
from sageworks.web_components.plugins import license_details
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.license_manager import LicenseManager

# Register this page with Dash
register_page(
    __name__,
    path="/license",
    name="SageWorks - License",
)

# Grab the API Key from the SageWorks ConfigManager
cm = ConfigManager()
api_key = cm.get_config("SAGEWORKS_API_KEY")
license_api_key = cm.get_config("LICENSE_API_KEY")
my_license_info = LicenseManager.load_api_license(aws_account_id=None, api_key=api_key, license_api_key=license_api_key)

# Put the components into 'dark' mode
load_figure_template("darkly")

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
        html.H2("SageWorks: License"),
        details_component,
    ],
    style={"padding": "12px 30px 30px 30px"},
)
