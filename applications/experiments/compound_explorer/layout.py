"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# FIXME
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode

m = Chem.MolFromSmiles("O=C1Nc2cccc3cccc1c23")
dos = Draw.MolDrawOptions()
SetDarkMode(dos)
dos.setBackgroundColour((0, 0, 0, 0))


def compound_explorer_layout(compound_scatter_plot: dcc.Graph) -> html.Div:
    """Set up the layout for the Compound Explorer Page"""
    layout = html.Div(
        dbc.Container(
            fluid=True,
            className="dbc dbc-ag-grid",
            style={"margin": "30px"},
            children=[
                # Page Header and Last Updated Timer
                dbc.Row(
                    [
                        html.H2("Compound Explorer (Alpha)"),
                        html.Div(
                            "Last Updated: ",
                            id="last-updated-compound-explorer",
                            style={
                                "color": "rgb(140, 200, 140)",
                                "fontSize": 15,
                                "padding": "0px 0px 0px 160px",
                            },
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
                            width=8,
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
                            width=4,
                            id="compound_diagram",
                        ),
                    ],
                ),
                html.Button("Update Plugin", id="update-button"),
            ],
        )
    )
    return layout
