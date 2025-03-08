"""Main Layout: Layout for the Main page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Workbench Imports
import workbench
from workbench.web_interface.components.plugins.ag_table import AGTable
from workbench.utils.config_manager import ConfigManager
from workbench.utils.theme_manager import ThemeManager

# Set the Main Page Title
tm = ThemeManager()
app_title = tm.branding().get("app_title", "Workbench Dashboard")


def main_layout(
    data_sources: AGTable,
    feature_sets: AGTable,
    models: AGTable,
    endpoints: AGTable,
    update_rate: int = 60,
) -> html.Div:
    """Main Layout for the Dashboard"""
    workbench_version = workbench.__version__.split("+")[0].strip()
    cm = ConfigManager()
    license_id = cm.get_license_id()

    # Update rate is in seconds (convert to milliseconds)
    update_rate = update_rate * 1000

    # Define the layout with one table per row
    layout = html.Div(
        children=[
            # This refreshes the page every 60 seconds
            dcc.Interval(id="main_page_refresh", interval=update_rate, n_intervals=0),
            dcc.Store(
                id="table_hashes",
                data={
                    "data_sources": None,
                    "feature_sets": None,
                    "models": None,
                    "endpoints": None,
                },
            ),
            # Top of Main Page Header/Info Section
            dbc.Row(
                [
                    html.Div(
                        [
                            html.H2(
                                [
                                    html.A(
                                        app_title + " ",
                                        href="/status",
                                        style={
                                            "textDecoration": "none",
                                            "color": "inherit",
                                        },
                                    ),
                                    html.Span(
                                        f"{workbench_version}",
                                        className="orange-text",
                                        style={"fontSize": 15},
                                    ),
                                    html.A(
                                        f"  [{license_id}]",
                                        href="/license",
                                        className="pink-text",
                                        style={
                                            "fontSize": 15,
                                            "textDecoration": "none",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "0px"},
                            ),
                            html.Div(
                                "Last Updated: ",
                                id="data-last-updated",
                                className="green-text",
                                style={
                                    "fontSize": 15,
                                    "fontStyle": "italic",
                                    "fontWeight": "bold",
                                    "paddingLeft": "200px",
                                },
                            ),
                        ]
                    )
                ]
            ),
            # Plugin Page Links
            dbc.Row(
                html.Div(
                    [
                        html.H4("Plugin Pages", id="plugin-pages-header", style={"textAlign": "left"}),
                        html.Ul([], id="plugin-pages-list"),  # Placeholder for the list
                    ],
                    id="plugin-pages",
                ),
                style={"padding": "20px 0px"},
            ),
            # Each table in its own row
            dbc.Row(
                [
                    html.H3("Data Sources", style={"textAlign": "left"}),
                    data_sources,
                ],
                style={"padding": "0px 0px"},
            ),
            dbc.Row(
                [
                    html.H3("Feature Sets", style={"textAlign": "left"}),
                    feature_sets,
                ],
                style={"padding": "20px 0px"},
            ),
            dbc.Row(
                [
                    html.H3("Models", style={"textAlign": "left"}),
                    models,
                ],
                style={"padding": "20px 0px"},
            ),
            dbc.Row(
                [
                    html.H3("Endpoints", style={"textAlign": "left"}),
                    endpoints,
                ],
                style={"padding": "20px 0px"},
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
