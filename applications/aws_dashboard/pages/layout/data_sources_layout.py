"""DataSources Layout: Layout for the DataSources page in the Artifact Viewer"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def data_sources_layout(components: dict) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: DataSources (Alpha)"),
                    html.Div(
                        "Last Updated: ",
                        id="last-updated-data-sources",
                        style={
                            "color": "rgb(140, 200, 140)",
                            "fontSize": 15,
                            "padding": "0px 0px 0px 160px",
                        },
                    ),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            # List out all the Data Sources
            dbc.Row(components["data_sources_table"]),
            # Data Source Details, Sample Rows, and Violin Plots
            # Row [ Sample Rows ]
            # Row [ Column 1                      Column 2 ]
            #       (Row(Data Source Details))    Row(Violin Plots)
            #
            dbc.Row(
                html.H3("Sampled Rows", id="sample_rows_header"),
                style={"padding": "30px 0px 10px 0px"},
            ),
            dbc.Row(
                components["data_source_sample_rows"],
                style={"padding": "0px 0px 30px 0px"},
            ),
            dbc.Row(
                [
                    # Column 1: Data Source Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="data_details_header"),
                                style={"padding": "0px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                components["data_source_details"],
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Column 2: Sample Rows and Violin Plots
                    dbc.Col(
                        [
                            dbc.Row(components["violin_plot"]),
                        ],
                        width=8,
                    ),
                    # Just the auto updater
                    dcc.Interval(id="data-sources-updater", interval=5000, n_intervals=0),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
