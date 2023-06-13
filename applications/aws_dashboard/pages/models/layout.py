"""Layout for the models page"""
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def models_layout(
    models_table: dash_table.DataTable,
    model_details: dcc.Markdown,
    confusion_matrix: dcc.Graph,
    scatter_plot: dcc.Graph,
    feature_importance: dcc.Graph,
    feature_details: dcc.Markdown,
) -> html.Div:
    layout = html.Div(
        children=[
            dbc.Row(html.H2("SageWorks: Models (Alpha)")),
            html.Div(
                "Last Updated: ",
                id="last-updated-models",
                style={
                    "color": "rgb(140, 200, 140)",
                    "fontSize": 15,
                    "padding": "0px 0px 0px 160px",
                },
            ),
            dbc.Row(style={"padding": "30px 0px 0px 0px"}),
            dbc.Row(
                [
                    # Model Table and Model Details
                    dbc.Col(
                        [
                            dbc.Row(models_table),
                            dbc.Row(
                                html.H3("Model Details"),
                                style={"padding": "50px 0px 0px 20px"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(model_details),
                                    dbc.Col(confusion_matrix),
                                ]
                            ),
                            dbc.Row(scatter_plot),
                        ],
                        width=8,
                    ),
                    # Feature Importance and Details
                    dbc.Col(
                        [
                            dbc.Row(html.H3("Feature Importance")),
                            dbc.Row(feature_importance),
                            dbc.Row(
                                html.H3(
                                    "Feature Details",
                                    style={"padding": "20px 0px 0px 20px"},
                                )
                            ),
                            dbc.Row(
                                feature_details,
                                style={"padding": "0px 0px 0px 20px"},
                            ),
                        ],
                        width=4,
                    ),
                ]
            ),
            dcc.Interval(id="models-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )
    return layout
