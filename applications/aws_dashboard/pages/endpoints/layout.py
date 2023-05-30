"""Endpoints Layout: Layout for the Endpoints page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def endpoints_layout(
    endpoints_table: dash_table.DataTable,
    endpoint_traffic: dcc.Graph,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: Endpoints (Alpha)"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated-endpoints",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 160px",
                        },
                    ),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            dbc.Row(endpoints_table),
            dbc.Row(endpoint_traffic),
            dcc.Interval(id="endpoints-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )
    return layout
