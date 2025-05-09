"""Layout for the Endpoints page"""

from typing import Any
from dash import html, dcc
import dash_bootstrap_components as dbc

from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.utils.theme_manager import ThemeManager
from workbench.utils.config_manager import ConfigManager

# Get the UI update rate
update_rate = ConfigManager().ui_update_rate() * 1000  # Convert to Milliseconds

# Get the Project Name
tm = ThemeManager()
project_name = tm.branding().get("project_name", "Workbench")


def endpoints_layout(
    endpoints_table: AGTable,
    endpoint_details: dcc.Markdown,
    endpoint_metrics: dcc.Graph,
    **kwargs: Any,
) -> html.Div:
    # Generate rows for each plugin
    plugin_rows = [
        dbc.Row(
            plugin,
            style={"padding": "0px 0px 0px 0px"},
        )
        for component_id, plugin in kwargs.items()
    ]
    layout = html.Div(
        children=[
            dcc.Interval(id="endpoints_refresh", interval=update_rate),
            dcc.Store(id="endpoints_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2(f"{project_name}: Endpoints"),
                    html.Div(id="dev_null", style={"display": "none"}),  # Output for callbacks without outputs
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(endpoints_table),
            # Model Details, and Plugins
            dbc.Row(
                [
                    # Column 1: Endpoint Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="endpoint_details_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                endpoint_details,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                        className="text-break",
                    ),
                    # Column 2: Endpoint Metrics and Plugins
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Endpoint Metrics", id="endpoint_metrics_header"),
                                style={"padding": "30px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                endpoint_metrics,
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                html.H3("Plugins", id="plugins_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            # Add the dynamically generated Plugin rows
                            *plugin_rows,
                        ],
                        width=8,
                    ),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
