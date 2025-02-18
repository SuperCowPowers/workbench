"""Layout for the Pipelines page"""

from typing import Any
from dash import html, dcc
import dash_bootstrap_components as dbc

from workbench.web_interface.components.plugins.ag_table import AGTable


def pipelines_layout(
    pipelines_table: AGTable,
    pipeline_details: dcc.Markdown,
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
            dcc.Interval(id="pipelines_refresh", interval=60000),
            dcc.Store(id="pipelines_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2("Workbench: Pipelines"),
                    html.Div(id="dev_null", style={"display": "none"}),  # Output for callbacks without outputs
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(pipelines_table),
            # Model Details, and Plugins
            dbc.Row(
                [
                    # Column 1: Model Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="pipeline_details_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                pipeline_details,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                        className="text-break",
                    ),
                    # Column 2: Pipeline Metrics and Plugins
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Pipeline Metrics", id="pipeline_metrics_header"),
                                style={"padding": "30px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                html.H3("TBD Plots", id="tbd_plots"),
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
