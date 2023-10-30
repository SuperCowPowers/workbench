"""Layout for the Models page"""
from typing import Any
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def models_layout(
    models_table: dash_table.DataTable,
    model_details: dcc.Markdown,
    model_metrics: dcc.Graph,
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
            dbc.Row(
                [
                    html.H2("SageWorks: Models"),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(models_table),
            # Model Details, and Plugins
            dbc.Row(
                [
                    # Column 1: Model Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="model_details_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                model_details,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Column 2: Model Metrics and Plugins
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Model Testing", id="model_metrics_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                model_metrics,
                                style={"padding": "30px 0px 10px 0px"},
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
