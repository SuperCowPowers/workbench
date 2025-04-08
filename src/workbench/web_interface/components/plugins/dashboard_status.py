"""A Markdown Component for details/information about the status of the Workbench Dashboard"""

import logging

# Dash Imports
from dash import html, dcc

# Workbench Imports
from workbench.web_interface.components.plugin_interface import PluginInterface, PluginPage, PluginInputType
from workbench.utils.workbench_cache import WorkbenchCache

# Get the Workbench logger
log = logging.getLogger("workbench")


class DashboardStatus(PluginInterface):
    """License Details Markdown Component"""

    # Initialize this Plugin Component Class with required attributes
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
                html.H3(id=f"{component_id}-header", children="Status: Loading..."),
                dcc.Markdown(id=f"{component_id}-details", children="Waiting for Data...", dangerously_allow_html=True),
                html.H3(id=f"{component_id}-support-header", children="Support: Loading..."),
                dcc.Markdown(
                    id=f"{component_id}-support-details", children="Waiting for Data...", dangerously_allow_html=True
                ),
            ],
        )

        # Fill in plugin properties
        self.properties = [
            (f"{component_id}-header", "children"),
            (f"{component_id}-details", "children"),
            (f"{component_id}-support-header", "children"),
            (f"{component_id}-support-details", "children"),
        ]

        # Return the container
        return container

    def update_properties(self, config_info: dict, **kwargs) -> list:
        """Update the properties for the plugin.

        Args:
            config_info (dict): A Workbench Configuration dictionary
            **kwargs: Additional keyword arguments (unused)

        Returns:
            list: A list of the updated property values for the plugin
        """
        # Header will just be blank (to give some space)
        header = ""

        # See if we can connect to the Redis Server
        if WorkbenchCache().check():
            details = "**Redis:** 🟢 Connected<br>"
        else:
            details = "**Redis:** 🔴 Failed to Connect<br>"

        # Fill in the license details
        details += f"**Redis Server:** {config_info['REDIS_HOST']}:{config_info.get('REDIS_PORT', 6379)}<br>"
        details += f"**Workbench S3 Bucket:** {config_info['WORKBENCH_BUCKET']}<br>"
        details += f"**Plugin Path:** {config_info.get('WORKBENCH_PLUGINS', 'unknown')}<br>"
        details += f"**Themes Path:** {config_info.get('WORKBENCH_THEMES', 'unknown')}<br>"
        details += f"**UI Update Rate:** {config_info.get('UI_UPDATE_RATE', 'unknown')}<br>"
        details += "**Workbench API Key:**\n"
        for key, value in config_info["API_KEY_INFO"].items():
            details += f"  - **{key}:** {value}\n"

        # Fill in the support details
        support_header = "Support Information"
        support_details = "- **Email:** [support@supercowpowers.com](mailto:support@supercowpowers.com)\n"
        support_details += "- **Chat:** [Discord](https://discord.gg/WHAJuz8sw8)\n"

        # Return the updated property values for the plugin
        return [header, details, support_header, support_details]


if __name__ == "__main__":
    # This class takes in license details and generates a details Markdown component
    import dash
    from workbench.utils.config_manager import ConfigManager

    # Just a quick unit test to make sure the component is working
    status_details = DashboardStatus()
    component = status_details.create_component("status_details")
    updated_properties = status_details.update_properties(ConfigManager().get_all_config())

    # Initialize Dash app
    app = dash.Dash(__name__)

    # Set the properties directly on the server side before initializing the layout
    for (component_id, prop), value in zip(status_details.properties, updated_properties):
        for child in component.children:
            if child.id == component_id:
                setattr(child, prop, value)
                break

    # Set the layout and run the app
    app.layout = html.Div([component])
    app.run(debug=True)
