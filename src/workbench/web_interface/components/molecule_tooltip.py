"""Molecule hover tooltip for Dash scatter plots and graphs."""

import logging

from dash import html

# Workbench Imports
from workbench.utils.chem_utils.vis import svg_from_smiles
from workbench.utils.color_utils import is_dark

log = logging.getLogger("workbench")


def molecule_hover_tooltip(
    smiles: str, mol_id: str = None, width: int = 300, height: int = 200, background: str = None
) -> list:
    """Generate a molecule hover tooltip from a SMILES string.

    This function creates a visually appealing tooltip with a dark background
    that displays the molecule ID at the top and structure below when hovering
    over scatter plot points.

    Args:
        smiles: SMILES string representing the molecule
        mol_id: Optional molecule ID to display at the top of the tooltip
        width: Width of the molecule image in pixels (default: 300)
        height: Height of the molecule image in pixels (default: 200)
        background: Optional background color (if None, uses dark gray)

    Returns:
        list: A list containing an html.Div with the ID header and molecule SVG,
              or an html.Div with an error message if rendering fails
    """
    # Use provided background or default to dark gray
    if background is None:
        background = "rgba(64, 64, 64, 1)"

    # Generate the SVG image from SMILES (base64 encoded data URI)
    img = svg_from_smiles(smiles, width, height, background=background)

    if img is None:
        log.warning(f"Could not render molecule for SMILES: {smiles}")
        return [
            html.Div(
                "Invalid SMILES",
                className="custom-tooltip",
                style={
                    "padding": "10px",
                    "color": "rgb(255, 140, 140)",
                    "width": f"{width}px",
                    "height": f"{height}px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            )
        ]

    # Build the tooltip with ID header and molecule image
    children = []

    # Add ID header if provided
    if mol_id is not None:
        # Set text color based on background brightness
        text_color = "rgb(200, 200, 200)" if is_dark(background) else "rgb(60, 60, 60)"
        children.append(
            html.Div(
                str(mol_id),
                style={
                    "textAlign": "center",
                    "padding": "8px",
                    "color": text_color,
                    "fontSize": "14px",
                    "fontWeight": "bold",
                    "borderBottom": "1px solid rgba(128, 128, 128, 0.5)",
                },
            )
        )

    # Add molecule image
    children.append(
        html.Img(
            src=img,
            style={"padding": "0px", "margin": "0px", "display": "block"},
            width=str(width),
            height=str(height),
        )
    )

    return [
        html.Div(
            children,
            className="custom-tooltip",
            style={"padding": "0px", "margin": "0px"},
        )
    ]


if __name__ == "__main__":
    from dash import Dash

    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div("Tooltip Preview:", style={"color": "white", "marginBottom": "20px"}),
            *molecule_hover_tooltip(aspirin, mol_id="Aspirin", background="rgba(200, 30, 30, 1)"),
        ],
        style={"background": "#1a1a1a", "padding": "50px"},
    )
    app.run(debug=True)
