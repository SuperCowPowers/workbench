"""Layout for the models page"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def models_layout(components: dict) -> html.Div:
    layout = html.Div(
        children=[
            dbc.Row(html.H2("SageWorks: Models (Mockup)")),
            html.Div(
                "Last Updated: ",
                id="last-updated-models",
                style={
                    "color": "rgb(140, 200, 140)",
                    "fontSize": 15,
                    "padding": "0px 0px 0px 160px",
                },
            ),
            dcc.Interval(id="models-updater", interval=10000, n_intervals=0),
            dbc.Row(style={"padding": "30px 0px 0px 0px"}),
            dbc.Row(
                [
                    # Model Table and Model Details
                    dbc.Col(
                        [
                            dbc.Row(components["models_table"]),
                            dbc.Row(
                                html.H3("Model Details"),
                                style={"padding": "50px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(components["model_details"]),
                                    dbc.Col(components["confusion_matrix"]),
                                ]
                            ),
                            dbc.Row(components["scatter_plot"]),
                        ],
                        width=8,
                    ),
                    # Feature Importance and Details
                    dbc.Col(
                        [
                            dbc.Row(html.H3("Feature Importance")),
                            dbc.Row(components["feature_importance"]),
                            dbc.Row(
                                html.H3(
                                    "Feature Details",
                                    style={"padding": "10px 0px 0px 0px"},
                                )
                            ),
                            dbc.Row(components["feature_details"]),
                        ],
                        width=4,
                    ),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
