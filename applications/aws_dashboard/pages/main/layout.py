"""Main Layout: Layout for the Main page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


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
            dcc.Store(data="", id="remove-artifact-store", storage_type="session"),
            dcc.Store(data="", id="modal-trigger-state-store", storage_type="session"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Attention")),
                    dbc.ModalBody(id="modal-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button("No", id="no-button", n_clicks=0),
                            dbc.Button("Yes", id="yes-button", n_clicks=0),
                        ]
                    ),
                ],
                id="modal",
                is_open=False,
            ),
            dbc.Row(
                [
                    html.H2("SageWorks Dashboard"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated",
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
            dcc.Interval(id="main-updater", interval=5000, n_intervals=0),
        ],
        style={"margin": "30px"},
    )
    return layout
