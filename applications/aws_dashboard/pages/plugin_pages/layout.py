"""Layout for the Models page"""
from typing import Any
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def plugin_layout(
    plugin_table: dash_table.DataTable,
    metrics_comparison_1: dcc.Markdown,
    metrics_comparison_2: dcc.Markdown,
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
            # Column 1: First model
            # TODO Display abbreviated model details on the same row
            dbc.Row(html.H2(id="model_header_1", style={"textAlign": "center"})),
            dbc.Row([dbc.Col([metrics_comparison_1], width=4), dbc.Col(model_metrics_1, width=8)]),
            # Column 2: Second model
            # TODO Display abbreviated model details on the same row
            dbc.Row(html.H2(id="model_header_2", style={"textAlign": "center"})),
            dbc.Row([dbc.Col([metrics_comparison_2], width=4), dbc.Col(model_metrics_2, width=8)]),
        ]
    )
    return layout
