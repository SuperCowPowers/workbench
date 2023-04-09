"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def feature_sets_layout(components: dict) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: FeatureSets (Alpha)"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated-feature-sets",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 160px",
                        },
                    ),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            dbc.Row(components["feature_sets_details"]),
            dbc.Row(
                [
                    dbc.Col(components["scatter1"]),
                    dbc.Col(components["scatter2"]),
                ]
            ),
            dcc.Interval(id="feature-sets-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )
    return layout
