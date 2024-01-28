"""DataSources Layout: Layout for the DataSources page in the Artifact Viewer"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def data_sources_layout(
    data_sources_table: dash_table.DataTable,
    data_source_sample_rows: dash_table.DataTable,
    data_source_details: dcc.Markdown,
    violin_plot: dcc.Graph,
    correlation_matrix: dcc.Graph,
) -> html.Div:
    # The layout for the DataSources page
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    html.H2("SageWorks: DataSources"),
                    dbc.Row(style={"padding": "30px 0px 0px 0px"}),
                ]
            ),
            # A table that lists out all the Data Sources
            dbc.Row(data_sources_table),
            # Sample/Outlier Rows for the selected Data Source AND Outlier Plot
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Sampled Rows", id="sample_rows_header"),
                                style={"padding": "30px 0px 10px 0px"},
                            ),
                            dbc.Row(
                                data_source_sample_rows,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ]
                    ),
                ]
            ),
            # Column1: Data Source Details, Column2: Violin Plots, Correlation Matrix
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
                                data_source_details,
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Column 2: Violin Plots (Correlation Matrix + Outliers)
                    dbc.Col(
                        [
                            dbc.Row(violin_plot),
                            dbc.Row(
                                [dbc.Col(correlation_matrix, width=12)],
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=8,
                    ),
                ]
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
