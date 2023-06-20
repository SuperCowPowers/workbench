"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def anomaly_layout(
    feature_sets_table: dash_table.DataTable,
    anomaly_table: dash_table.DataTable,
    scatter_plot: dcc.Graph,
    violin_plot: dcc.Graph,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: DNS Anomaly Inspector"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated-anomaly-table",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 160px",
                        },
                    ),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            dbc.Row(feature_sets_table),
            dbc.Row(
                [
                    dbc.Col(
                        [anomaly_table],
                        style={"padding": "20px 5px 10px 0px"},
                        width=6,
                    ),
                    dbc.Col(scatter_plot, width=6, style={"padding": "30px 0px 10px 5px"},),
                ]
            ),
            dbc.Row(violin_plot),
            dcc.Interval(id="anomaly-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )

    return layout

