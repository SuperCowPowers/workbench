"""A Markdown Component for details/information about a SageWorks License"""

import logging

# Dash Imports
from dash import html, dcc

# SageWorks Imports
from sageworks.web_components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from sageworks.utils.license_manager import LicenseManager

# Get the SageWorks logger
log = logging.getLogger("sageworks")


class LicenseDetails(PluginInterface):
    """License Details Markdown Component"""

    """Initialize this Plugin Component Class with required attributes"""
    auto_load_page = PluginPage.NONE
    plugin_input_type = PluginInputType.CUSTOM

    def create_component(self, component_id: str) -> html.Div:
        """Create a Markdown Component without any data.
        Args:
            component_id (str): The ID of the web component
        Returns:
            html.Div: A Container of Components for the Model Details
        """
        container = html.Div(
            id=component_id,
            children=[
                html.H3(id=f"{component_id}-header", children="License: Loading..."),
                dcc.Markdown(id=f"{component_id}-details", children="Waiting for Data...", dangerously_allow_html=True),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{component_id}-header", "children"),
            (f"{component_id}-details", "children"),
        ]

        # Return the container
        return container

    def update_properties(self, license: dict, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            license (dict): A SageWorks License dictionary
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        # Update the header and the details
        header = "License Details"

        # See if we can connect to the License Server
        response = LicenseManager.contact_license_server()
        if response.status_code == 200:
            details = "**License Server:** Connected<br>"
        else:
            # Note: This is 100% fine/expected (connecting to the license server is optional)
            details = "**License Server:** Not connected<br>"

        # Fill in the license details
        details += f"**License Id:** {license['license_id']}<br>"
        details += f"**Company:** {license['license_id']}<br>"
        details += f"**AWS Account:** {license['aws_account_id']}<br>"
        details += f"**Expiration:** {license['expires']}<br>"
        details += f"**License Tier:** {license.get('tier', 'Open Source')}<br>"
        details += "**Features:**\n"
        for feature, value in license["features"].items():
            details += f"  - **{feature}:** {value}\n"

        # Return the updated property values for the plugin
        return [header, details]


if __name__ == "__main__":
    # This class takes in license details and generates a details Markdown component
    import dash
    from sageworks.utils.config_manager import ConfigManager

    # Grab the API Key from the SageWorks ConfigManager
    cm = ConfigManager()
    api_key = cm.get_config("SAGEWORKS_API_KEY")
    my_license_info = LicenseManager.load_api_license(aws_account_id=None, api_key=api_key)

    # Just a quick unit test to make sure the component is working
    license_details = LicenseDetails()
    component = license_details.create_component("license_details")
    updated_properties = license_details.update_properties(my_license_info)

    # Initialize Dash app
    app = dash.Dash(__name__)

    # Set the properties directly on the server side before initializing the layout
    for (component_id, prop), value in zip(license_details.properties, updated_properties):
        for child in component.children:
            if child.id == component_id:
                setattr(child, prop, value)
                break

    # Set the layout and run the app
    app.layout = html.Div([component])
    app.run(debug=True)
