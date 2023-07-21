"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def data_sources_layout(
    data_sources_table: dash_table.DataTable,
    data_source_outlier_rows: dash_table.DataTable,
    outlier_scatter_plot: dcc.Graph,
    data_source_details: dcc.Markdown,
    violin_plot: dcc.Graph,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: Outlier Inspector (Alpha)"),
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
            dbc.Row(data_sources_table),
            # Data Source Details, Outlier Rows, Scatter Plot, and Violin Plots
            # Row [ Column 1                      Column 2 ]
            #       (Row(Outlier Rows))           Row(Cluster/Scatter Plot)
            # Row [ Column 1                      Column 2 ]
            #       (Row(Data Source Details))    Row(Violin Plots)
            #
            dbc.Row(
                html.H3("Outlier Rows", id="data_source_outlier_rows_header"),
                style={"padding": "30px 0px 10px 0px"},
            ),
            dbc.Row(
                [
                    # Column 1: Outlier Rows
                    dbc.Col(
                        [
                            dbc.Row(
                                data_source_outlier_rows,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=6,
                    ),
                    # Column 2: Cluster/Scatter Plot
                    dbc.Col(
                        [
                            dbc.Row(
                                outlier_scatter_plot,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    # Column 1: Data Source Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="data_source_details_header"),
                                style={"padding": "0px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                data_source_details,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Column 2: Sample Rows and Violin Plots
                    dbc.Col(
                        [
                            dbc.Row(violin_plot),
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
