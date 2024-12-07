"""Main Layout: Layout for the Main page in the Artifact Viewer"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# Get the SageWorks Version and License ID
import sageworks
from sageworks.utils.config_manager import ConfigManager


def main_layout(
    data_sources: dash_table.DataTable,
    feature_sets: dash_table.DataTable,
    models: dash_table.DataTable,
    endpoints: dash_table.DataTable,
    update_rate: int = 60,
) -> html.Div:
    """Main Layout for the Dashboard"""
    sageworks_version = sageworks.__version__.split("+")[0].strip()
    cm = ConfigManager()
    license_id = cm.get_license_id()

    # Update rate is in seconds (convert to milliseconds)
    update_rate = update_rate * 1000

    # Define the layout with 2 rows and 2 columns
    layout = html.Div(
        children=[
            dcc.Interval(id="main_page_refresh", interval=update_rate, n_intervals=0),
            # Header Section
            dbc.Row(
                [
                    html.H2(
                        [
                            html.A(
                                "SageWorks Dashboard ",
                                href="/status",
                                style={"color": "rgb(200, 200, 200)", "textDecoration": "none"},
                            ),
                            html.Span(
                                f"{sageworks_version}",
                                style={
                                    "color": "rgb(180, 120, 180)",
                                    "fontSize": 15,
                                },
                            ),
                            html.Span(
                                html.A(
                                    f"  [{license_id}]",
                                    href="/license",
                                    style={
                                        "color": "rgb(140, 140, 200)",
                                        "fontSize": 15,
                                        "textDecoration": "none",
                                    },
                                ),
                            ),
                        ]
                    ),
                    html.Div(
                        "Last Updated: ",
                        id="data-last-updated",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 120px",
                        },
                    ),
                ]
            ),
            # First row with 2 columns
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Data Sources", style={"textAlign": "center"}),
                            data_sources,
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H3("Feature Sets", style={"textAlign": "center"}),
                            feature_sets,
                        ],
                        width=6,
                    ),
                ],
                style={"padding": "20px 0px"},
            ),
            # Second row with 2 columns
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Models", style={"textAlign": "center"}),
                            models,
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H3("Endpoints", style={"textAlign": "center"}),
                            endpoints,
                        ],
                        width=6,
                    ),
                ],
                style={"padding": "20px 0px"},
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
