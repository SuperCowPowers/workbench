"""Layout for the Models page"""
from typing import Any
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def plugin_layout(
    plugin_table: dash_table.DataTable,
    model_metrics_1: dcc.Graph,
    model_metrics_2: dcc.Graph,
    **kwargs: Any,
) -> html.Div:
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: Plugin Page"),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(plugin_table),
            # Model metrics
            dbc.Row(
                [
                    # Column 1: First model
                    #TODO Display abbreviated model details on the same row
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Model 1", id="model_1_header"),
                                style={"padding": "30px 0px 10px 0px"}
                            ),
                            dbc.Row(
                                model_metrics_1,
                                style={"padding": "0px 0px 30px 0px"},
                            )
                        ],
                        width=10,
                    ),
                    # Column 2: Second model
                    #TODO Display abbreviated model details on the same row
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Model 2", id="model_2_header"),
                                style={"padding": "30px 0px 10px 0px"}
                            ),
                            dbc.Row(
                                model_metrics_2,
                                style={"padding": "0px 0px 30px 0px"},
                            )
                        ],
                        width=10,
                    )
                ]
            )
        ],
        style={"margin": "30px"},
    )
    return layout
