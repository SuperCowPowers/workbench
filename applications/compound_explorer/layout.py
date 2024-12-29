"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def compound_explorer_layout(scatter_plot: dcc.Graph, molecule_view: html.Div) -> html.Div:
    """Set up the layout for the Compound Explorer Page"""
    layout = html.Div(
        dbc.Container(
            fluid=True,
            className="dbc dbc-ag-grid",
            style={"margin": "30px"},
            children=[
                # Page Header and Last Updated Timer
                dbc.Row(
                    [
                        html.H2("Compound Explorer"),
                        html.Div(
                            "Last Updated: ",
                            id="last-updated-compound-explorer",
                            style={
                                "color": "rgb(140, 200, 140)",
                                "fontSize": 15,
                                "padding": "0px 0px 0px 160px",
                            },
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        # Column 1: Scatter Plot
                        dbc.Col([scatter_plot], width=8),
                        # Column 2: Molecular Viewer
                        dbc.Col([molecule_view], width=4),
                    ],
                ),
                html.Button("Update Plugin", id="update-button"),
            ],
        )
    )
    return layout
