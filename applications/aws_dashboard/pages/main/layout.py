"""Main Layout: Layout for the Main page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# Get the SageWorks Version
import sageworks

sageworks_version = sageworks.__version__


def main_layout(
    incoming_data: dash_table.DataTable,
    glue_jobs: dash_table.DataTable,
    data_sources: dash_table.DataTable,
    feature_sets: dash_table.DataTable,
    models: dash_table.DataTable,
    endpoints: dash_table.DataTable,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dcc.Interval(id="broker-update-timer", interval=5000, n_intervals=0),
            dbc.Row(
                [
                    html.H2(
                        [
                            "SageWorks Dashboard ",
                            html.Span(
                                f" v {sageworks_version}",
                                style={
                                    "color": "rgb(200, 140, 200)",
                                    "fontSize": 15,
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        "Last Updated: ",
                        id="data-last-updated",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 120px",
                        },
                    ),
                ]
            ),
            dbc.Row(html.H3("Incoming Data"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(incoming_data),
            dbc.Row(html.H3("Glue Jobs"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(glue_jobs),
            dbc.Row(html.H3("Data Sources"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(data_sources),
            dbc.Row(html.H3("Feature Sets"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(feature_sets),
            dbc.Row(html.H3("Models"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(models),
            dbc.Row(html.H3("Endpoints"), style={"padding": "10px 0px 0px 0px"}),
            dbc.Row(endpoints),
        ],
        style={"margin": "30px"},
    )
    return layout
