"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def feature_sets_layout(
    feature_sets_table: dash_table.DataTable,
    feature_set_anomalies_rows: dash_table.DataTable,
    anomaly_scatter_plot: dcc.Graph,
    feature_set_details: dcc.Markdown,
    violin_plot: dcc.Graph,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: Anomaly Inspector (Alpha)"),
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
            # List out all the Data Sources
            dbc.Row(feature_sets_table),
            # Data Source Details, Anomalous Rows, Scatter Plot, and Violin Plots
            # Row [ Column 1                      Column 2 ]
            #       (Row(Anomalous Rows))         Row(Cluster/Scatter Plot)
            # Row [ Column 1                      Column 2 ]
            #       (Row(Data Source Details))    Row(Violin Plots)
            #
            dbc.Row(
                html.H3("Sampled Rows", id="feature_sample_rows_header"),
                style={"padding": "30px 0px 10px 0px"},
            ),
            dbc.Row(
                [
                    # Column 1: Anomalous Rows
                    dbc.Col(
                        [
                            dbc.Row(
                                feature_set_anomalies_rows,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=8,
                    ),
                    # Column 2: Cluster/Scatter Plot
                    dbc.Col(
                        [
                            dbc.Row(
                                anomaly_scatter_plot,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    # Column 1: Data Source Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="feature_details_header"),
                                style={"padding": "0px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                feature_set_details,
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
                    dcc.Interval(id="feature-sets-updater", interval=5000, n_intervals=0),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
