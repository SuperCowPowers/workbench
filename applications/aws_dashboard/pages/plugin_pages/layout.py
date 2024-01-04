"""Layout for the Models page"""
from typing import Any
from dash import html, dash_table
import dash_bootstrap_components as dbc


def plugin_layout(
    plugin_table: dash_table.DataTable,
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
            dbc.Row(plugin_table),
            # Add the dynamically generated Plugin rows
            *plugin_rows,
        ],
        style={"margin": "30px"},
    )
    return layout
