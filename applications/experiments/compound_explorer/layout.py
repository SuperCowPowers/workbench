"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# FIXME
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode

m = Chem.MolFromSmiles("O=C1Nc2cccc3cccc1c23")
dos = Draw.MolDrawOptions()
SetDarkMode(dos)
dos.setBackgroundColour((0, 0, 0, 0))
# Draw.MolToImage(m, 'foo.png')


def data_sources_layout(
    data_sources_table: dash_table.DataTable,
    compound_rows: dash_table.DataTable,
    compound_scatter_plot: dcc.Graph,
    data_source_details: dcc.Markdown,
    violin_plot: dcc.Graph,
) -> html.Div:
    # Just put all the tables in as Rows for Now (do something fancy later)
    layout = html.Div(
        children=[
            # Page Header and Last Updated Timer
            dbc.Row(
                [
                    html.H2("SageWorks: Compounds Explorer (Alpha)"),
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
            dbc.Row(
                [
                    # Column 1: Data Source Details
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H3("Details", id="data_source_details_header"),
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                data_source_details,
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Column 2: Data Sources Table
                    dbc.Col(
                        [
                            dbc.Row(
                                data_sources_table,
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=8,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            # Cluster/Scatter Plot
                            dbc.Row(compound_scatter_plot),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            # Violin Plot
                            dbc.Row(
                                violin_plot,
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=8,
                    ),
                ]
            ),
            dbc.Row(
                html.H3("Compounds", id="data_source_rows_header"),
                style={"padding": "0px 0px 0px 0px"},
            ),
            dbc.Row(
                [
                    # Column 1: Compound Table (Data Source Rows)
                    dbc.Col(
                        [
                            dbc.Row(
                                compound_rows,
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=9,
                    ),
                    # Column 2: Compound Diagram
                    dbc.Col(
                        [
                            dbc.Row(
                                html.H5("<compound name>"),
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                html.Img(
                                    src=Draw.MolToImage(m, options=dos, size=(300, 300)),
                                    style={"height": "300", "width": "300"},
                                ),
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                            dbc.Row(
                                html.H5("<compound info>"),
                                style={"padding": "0px 0px 0px 0px"},
                            ),
                        ],
                        width=3,
                        id="compound_diagram",
                    ),
                ],
            ),
        ],
        style={"margin": "30px"},
    )
    return layout
