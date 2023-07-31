"""FeatureSets Layout: Layout for the FeatureSets page in the Artifact Viewer"""
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# FIXME
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


m = Chem.MolFromSmiles('O=C1Nc2cccc3cccc1c23')
mol = rdMolDraw2D.PrepareMolForDrawing(m)
drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
drawer.DrawMolecule(mol)
drawer.FinishDrawing()
svg_xml = drawer.GetDrawingText()
with open('assets/mol_test.svg', 'w') as f:
    f.write(svg_xml)
# svg_xml = svg_xml.encode("utf-8")
# encoded = base64.b64encode(svg_xml)
# svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
# svg = 'data:image/svg+xml;base64,{}'.format(svg_xml)
# svg = 'data:image/svg+xml;{}'.format(svg_xml.decode())

# svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())

# Draw.MolToImage(m, 'foo.png')

                    dbc.Col(
                        [
                            dbc.Row(
                                #html.Img('data:image/svg,{}'.format(svg_xml), width=300, height=300),
                                html.Img(src="assets/mol_test.svg", width=300, height=300),
                                # html.Img(svg_xml),
                                style={"padding": "0px 0px 30px 0px"},
                            ),
                        ],
                        width=6,
                    ),
