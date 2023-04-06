"""Layout for the models page"""
from dash import Dash, html
import dash_bootstrap_components as dbc


def models_layout(components: dict) -> html.Div:
    layout = html.Div(
        children=[
            dbc.Row(html.H2('SageWorks: Model Details')),
            dbc.Row(
                [
                    # Model Table and Model Details
                    dbc.Col(
                        [
                            dbc.Row(components["model_table"]),
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
