"""Layout for the Models page"""

from typing import Any
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def models_layout(
    models_table: dash_table.DataTable,
    model_details: html.Div,
    model_plot: dcc.Graph,
    **kwargs: Any,
) -> html.Div:
    # Generate rows for each plugin
    plugin_rows = [dbc.Row(plugin, style={"padding": "0px 0px 20px 0px"}) for component_id, plugin in kwargs.items()]
    layout = html.Div(
        children=[
            dcc.Interval(id="models_refresh", interval=60000),
            dcc.Store(id="models_page_loaded", data=False),
            dbc.Row(
                [
                    html.H2("Workbench: Models"),
                ]
            ),
            # A table that lists out all the Models
            dbc.Row(models_table),
            # Model Details, and Plugins
            dbc.Row(
                [
                    # Column 1: Model Details
                    dbc.Col(model_details, width=4, style={"padding": "30px 0px 0px 0px"}, className="text-break"),
                    # Column 2: Model Plot and Plugins
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Performance", id="model_plot_header"),
                                style={"padding": "30px 0px 0px 0px"},
                            ),
                            dbc.Row(model_plot),
                            dbc.Row(
                                html.H3("Plugins", id="plugins_header"),
                                style={"padding": "30px 0px 0px 0px"},
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
