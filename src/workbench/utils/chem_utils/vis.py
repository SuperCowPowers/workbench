"""Molecular visualization utilities for Workbench"""

import logging
import base64
import sys
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from dash import html

# Workbench Imports
from workbench.utils.color_utils import is_dark

# Set up the logger
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
    try:

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

    except ImportError as e:
        log.error(f"RDKit not available for molecule rendering: {e}")
        return [
            html.Div(
                "RDKit not installed",
                className="custom-tooltip",
                style={
                    "padding": "10px",
                    "color": "rgb(255, 195, 140)",
                    "width": f"{width}px",
                    "height": f"{height}px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            )
        ]


def _rgba_to_tuple(rgba: str) -> Tuple[float, float, float, float]:
    """Convert rgba string to normalized tuple (R, G, B, A).

    Args:
        rgba: RGBA color string (e.g., "rgba(255, 0, 0, 0.5)")

    Returns:
        Normalized tuple of (R, G, B, A) with RGB in [0, 1]
    """
    try:
        components = rgba.strip("rgba() ").split(",")
        r, g, b = (int(components[i]) / 255 for i in range(3))
        a = float(components[3]) if len(components) > 3 else 1.0
        return r, g, b, a
    except (IndexError, ValueError) as e:
        log.warning(f"Error parsing color '{rgba}': {e}, using default")
        return 0.25, 0.25, 0.25, 1.0  # Default dark grey


def _validate_molecule(smiles: str) -> Optional[Chem.Mol]:
    """Validate and return RDKit molecule from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        RDKit molecule or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"Invalid SMILES: {smiles}")
        return mol
    except Exception as e:
        log.error(f"Error parsing SMILES '{smiles}': {e}")
        return None


def _configure_draw_options(options: Draw.MolDrawOptions, background: str) -> None:
    """Configure drawing options for molecule visualization.

    Args:
        options: RDKit drawing options object
        background: Background color string
    """
    try:
        if is_dark(background):
            rdMolDraw2D.SetDarkMode(options)
        # Light backgrounds use RDKit defaults (no action needed)
    except ValueError:
        # Default to dark mode if color format is invalid
        log.warning(f"Invalid color format: {background}, defaulting to dark mode")
        rdMolDraw2D.SetDarkMode(options)
    options.setBackgroundColour(_rgba_to_tuple(background))


def img_from_smiles(
    smiles: str, width: int = 500, height: int = 500, background: str = "rgba(64, 64, 64, 1)"
) -> Optional:
    """Generate an image of the molecule from SMILES.

    Args:
        smiles: SMILES string representing the molecule
        width: Width of the image in pixels (default: 500)
        height: Height of the image in pixels (default: 500)
        background: Background color (default: dark grey)

    Returns:
        PIL Image object or None if SMILES is invalid
    """
    mol = _validate_molecule(smiles)
    if not mol:
        return None

    # Set up drawing options
    dos = Draw.MolDrawOptions()
    _configure_draw_options(dos, background)

    # Generate and return image
    return Draw.MolToImage(mol, options=dos, size=(width, height))


def svg_from_smiles(
    smiles: str, width: int = 500, height: int = 500, background: str = "rgba(64, 64, 64, 1)"
) -> Optional[str]:
    """Generate an SVG image of the molecule from SMILES.

    Args:
        smiles: SMILES string representing the molecule
        width: Width of the image in pixels (default: 500)
        height: Height of the image in pixels (default: 500)
        background: Background color (default: dark grey)

    Returns:
        Base64-encoded SVG data URI or None if SMILES is invalid
    """
    mol = _validate_molecule(smiles)
    if not mol:
        return None

    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Initialize SVG drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

    # Configure drawing options
    _configure_draw_options(drawer.drawOptions(), background)

    # Draw molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Encode SVG as base64 data URI
    svg = drawer.GetDrawingText()
    encoded_svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded_svg}"


def show(smiles: str, width: int = 500, height: int = 500, background: str = "rgba(64, 64, 64, 1)") -> None:
    """Display an image of the molecule.

    Args:
        smiles: SMILES string representing the molecule
        width: Width of the image in pixels (default: 500)
        height: Height of the image in pixels (default: 500)
        background: Background color (default: dark grey)
    """
    img = img_from_smiles(smiles, width, height, background)
    if img:
        img.show()
    else:
        log.error(f"Cannot display molecule for SMILES: {smiles}")


def molecule_grid(
    smiles: list,
    captions: list = None,
    caption_colors: list = None,
    ncols: int = 3,
    mol_size: int = 400,
    suptitle: str = None,
):
    """Lay out molecule structures in a labeled matplotlib grid.

    Handles the grid mechanics -- sizing, axis-off, invalid-SMILES gaps, blank
    trailing cells -- and leaves the domain choices (what each caption says, what
    color it is) to the caller. Tiles render on a transparent background, so the grid
    drops cleanly onto any page. Good for top-residual panels and arbitrary structure
    sets where seeing the molecules side by side tells the story.

    Args:
        smiles (list[str]): One SMILES per molecule.
        captions (list[str], optional): Caption under each molecule (id, metrics).
            None for no captions.
        caption_colors (list[str], optional): Per-caption color (any matplotlib
            color, e.g. "gold", "#87d75f"). Defaults to black.
        ncols (int, optional): Columns in the grid. Defaults to 3.
        mol_size (int, optional): Rendered width of each molecule in pixels; the
            height is 2/3 of this. Defaults to 400.
        suptitle (str, optional): Figure-level title.

    Returns:
        matplotlib.figure.Figure: The grid figure. The caller shows or saves it:
            `fig.show()`, or `fig.savefig(path, dpi=150, bbox_inches="tight")`.
    """
    import matplotlib.pyplot as plt

    mol_height = round(mol_size * 2 / 3)
    n = len(smiles)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), constrained_layout=True)
    fig.patch.set_alpha(0.0)  # transparent figure background
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        ax.patch.set_alpha(0.0)
        if i >= n:
            continue  # blank trailing cell
        # White (non-dark) RGB keeps RDKit's dark bonds; alpha 0 makes the tile transparent
        img = img_from_smiles(smiles[i], width=mol_size, height=mol_height, background="rgba(255, 255, 255, 0)")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "invalid SMILES", ha="center", va="center", color="grey")
        if captions is not None:
            color = caption_colors[i] if caption_colors is not None else "black"
            ax.set_title(captions[i], color=color, fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    return fig


def neighborhood_graph(
    query_id,
    neighbors_df,
    target_col,
    smiles_col: str = "smiles",
    id_col: str = "neighbor_id",
    similarity_col: str = "similarity",
    n_neighbors: int = 5,
    cmap_name: str = "viridis",
    title: str = None,
):
    """Radial graph of a query compound and its nearest neighbors.

    The query sits at the center with the `n_neighbors` most-similar neighbors around
    a circle. Each node renders the molecule inside a colored ring (ring color = target
    value), and edge width scales with similarity -- so an activity cliff (near-identical
    structures, very different target) reads at a glance.

    Args:
        query_id: Id of the center compound; must appear in `neighbors_df[id_col]`.
        neighbors_df: One row per compound, with id, smiles, similarity, and target
            columns. Must include the query's own row -- e.g. from
            `neighbors(..., include_self=True)`, where its similarity is 1.0.
        target_col: Column holding the target value; drives ring color (NaN -> gray).
        smiles_col (str): Column holding SMILES strings. Defaults to "smiles".
        id_col (str): Column holding compound ids. Defaults to "neighbor_id".
        similarity_col (str): Column holding similarity to the query, higher = closer.
            Defaults to "similarity".
        n_neighbors (int): How many closest neighbors to draw. Defaults to 5.
        cmap_name (str): Matplotlib colormap for the target ring/colorbar. Defaults
            to "viridis".
        title (str, optional): Plot title. A sensible default is built when None.

    Returns:
        matplotlib.figure.Figure: The graph figure. The caller shows or saves it:
            `fig.show()`, or `fig.savefig(path, dpi=150, bbox_inches="tight")`.
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.cm import ScalarMappable

    df = neighbors_df.dropna(subset=[smiles_col]).copy()
    query_match = df[df[id_col] == query_id]
    if query_match.empty:
        raise ValueError(
            f"query id {query_id!r} not found in neighbors_df[{id_col!r}] -- include the "
            f"query's own row, e.g. neighbors(..., include_self=True)."
        )
    query_row = query_match.iloc[0]
    neigh = df[df[id_col] != query_id].nlargest(n_neighbors, similarity_col).reset_index(drop=True)
    ring_rows = pd.concat([query_row.to_frame().T, neigh], ignore_index=True)

    # Ring color scale over the query + drawn neighbors.
    vals = pd.to_numeric(ring_rows[target_col], errors="coerce")
    finite = vals.dropna()
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[cmap_name]

    def ring_color(v):
        return "gray" if pd.isna(v) else cmap(norm(v))

    # Circular molecule image (white background masked to a disc).
    def circle_img(smiles, px=340):
        img = img_from_smiles(smiles, width=px, height=px, background="rgba(255,255,255,1)")
        if img is None:
            return np.zeros((px, px, 4), dtype=np.uint8)
        arr = np.array(img.convert("RGBA"))
        yy, xx = np.ogrid[:px, :px]
        r = px / 2
        mask = (xx - r) ** 2 + (yy - r) ** 2 <= (r - 2) ** 2
        arr[~mask, 3] = 0
        return arr

    fig, ax = plt.subplots(figsize=(11.5, 11))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_autoscale_on(False)  # keep imshow from rescaling the axes

    center = (0.0, 0.0)
    angles = np.linspace(90, 90 - 360, len(neigh), endpoint=False)
    radius = 1.15
    positions = {query_id: center}
    for row, ang in zip(neigh.itertuples(), angles):
        positions[getattr(row, id_col)] = (radius * np.cos(np.radians(ang)), radius * np.sin(np.radians(ang)))

    # Edge width scaled across the observed similarity range.
    sims = neigh[similarity_col]
    smin, smax = (float(sims.min()), float(sims.max())) if len(sims) else (0.0, 1.0)

    def edge_lw(sim):
        frac = (sim - smin) / (smax - smin) if smax > smin else 1.0
        return 1.5 + 11 * frac

    for row in neigh.itertuples():
        x, y = positions[getattr(row, id_col)]
        sim = getattr(row, similarity_col)
        ax.plot([center[0], x], [center[1], y], color="#888", lw=edge_lw(sim), zorder=1, alpha=0.7)
        ax.text(
            x * 0.5,
            y * 0.5,
            f"{sim:.2f}",
            fontsize=11,
            color="#333",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", fc="white", ec="#888"),
            zorder=2,
        )

    node_r, gap = 0.40, 0.03
    for nid, (x, y) in positions.items():
        row = ring_rows[ring_rows[id_col] == nid].iloc[0]
        tval = pd.to_numeric(row[target_col], errors="coerce")
        ax.add_patch(Circle((x, y), node_r, facecolor="white", edgecolor=ring_color(tval), lw=6, zorder=3))
        ax.imshow(circle_img(row[smiles_col]), extent=(x - node_r, x + node_r, y - node_r, y + node_r), zorder=4)
        tstr = "n/a" if pd.isna(tval) else f"{tval:.2f}"
        lbl = "QUERY" if nid == query_id else "neighbor"
        va, off = ("bottom", node_r + gap) if y > 0.5 else ("top", -node_r - gap)
        ax.text(x, y + off, f"{nid}\n{lbl}  {target_col}={tstr}", fontsize=10.5, ha="center", va=va, zorder=5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
    cbar.set_label(target_col, fontsize=12)

    if title is None:
        title = (
            f"{query_id} + {len(neigh)} closest neighbors\n"
            f"(ring = {target_col}, edge width = {similarity_col})"
        )
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Test suite
    print("Running molecular visualization tests...")

    # Test molecules
    test_molecules = {
        "benzene": "c1ccccc1",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "invalid": "not_a_smiles",
        "empty": "",
    }

    # Test 1: Valid SMILES image generation
    print("\n1. Testing image generation from valid SMILES...")
    for name, smiles in test_molecules.items():
        if name not in ["invalid", "empty"]:
            img = img_from_smiles(smiles, width=200, height=200)
            status = "✓" if img else "✗"
            print(f"   {status} {name}: {'Success' if img else 'Failed'}")

    # Test 2: Invalid SMILES handling
    print("\n2. Testing invalid SMILES handling...")
    img = img_from_smiles(test_molecules["invalid"])
    print(f"   {'✓' if img is None else '✗'} Invalid SMILES returns None: {img is None}")

    img = img_from_smiles(test_molecules["empty"])
    print(f"   {'✓' if img is None else '✗'} Empty SMILES returns None: {img is None}")

    # Test 3: SVG generation
    print("\n3. Testing SVG generation...")
    for name, smiles in test_molecules.items():
        if name not in ["invalid", "empty"]:
            svg = svg_from_smiles(smiles, width=200, height=200)
            is_valid = svg and svg.startswith("data:image/svg+xml;base64,")
            status = "✓" if is_valid else "✗"
            print(f"   {status} {name}: {'Valid SVG data URI' if is_valid else 'Failed'}")

    # Test 4: Different backgrounds
    print("\n4. Testing different background colors...")
    backgrounds = [
        ("Light", "rgba(255, 255, 255, 1)"),
        ("Dark", "rgba(0, 0, 0, 1)"),
        ("Custom", "rgba(100, 150, 200, 0.8)"),
    ]

    for bg_name, bg_color in backgrounds:
        img = img_from_smiles(test_molecules["benzene"], background=bg_color)
        status = "✓" if img else "✗"
        print(f"   {status} {bg_name} background: {'Success' if img else 'Failed'}")

    # Test 5: Size variations
    print("\n5. Testing different image sizes...")
    sizes = [(100, 100), (500, 500), (1000, 800)]

    for w, h in sizes:
        img = img_from_smiles(test_molecules["caffeine"], width=w, height=h)
        status = "✓" if img else "✗"
        print(f"   {status} Size {w}x{h}: {'Success' if img else 'Failed'}")

    # Test 6: Color parsing functions
    print("\n6. Testing color utility functions...")
    test_colors = [
        ("invalid_color", None, (0.25, 0.25, 0.25, 1.0)),  # Should raise ValueError
        ("rgba(255, 255, 255, 1)", False, (1.0, 1.0, 1.0, 1.0)),
        ("rgba(0, 0, 0, 1)", True, (0.0, 0.0, 0.0, 1.0)),
        ("rgba(64, 64, 64, 0.5)", True, (0.251, 0.251, 0.251, 0.5)),
        ("rgb(128, 128, 128)", False, (0.502, 0.502, 0.502, 1.0)),
    ]

    for color, expected_dark, expected_tuple in test_colors:
        try:
            is_dark_result = is_dark(color)
            if expected_dark is None:
                print(f"   ✗ is_dark('{color[:20]}...'): Expected ValueError but got {is_dark_result}")
            else:
                dark_status = "✓" if is_dark_result == expected_dark else "✗"
                print(f"   {dark_status} is_dark('{color[:20]}...'): {is_dark_result} == {expected_dark}")
        except ValueError:
            if expected_dark is None:
                print(f"   ✓ is_dark('{color[:20]}...'): Correctly raised ValueError")
            else:
                print(f"   ✗ is_dark('{color[:20]}...'): Unexpected ValueError")

        tuple_result = _rgba_to_tuple(color)
        # Check tuple values with tolerance for floating point
        tuple_match = all(abs(a - b) < 0.01 for a, b in zip(tuple_result, expected_tuple))
        tuple_status = "✓" if tuple_match else "✗"
        print(f"   {tuple_status} rgba_to_tuple('{color[:20]}...'): matches expected")

    # Test 7: molecule_grid layout
    print("\n7. Testing molecule_grid...")
    import matplotlib

    matplotlib.use("Agg")  # headless
    grid_smiles = [test_molecules["benzene"], test_molecules["caffeine"], test_molecules["invalid"]]
    fig = molecule_grid(
        grid_smiles,
        captions=["benzene", "caffeine", "bad"],
        caption_colors=["gold", "#87d75f", "salmon"],
        ncols=2,
        suptitle="Test grid",
    )
    n_axes = len(fig.axes)
    print(f"   {'✓' if n_axes == 4 else '✗'} 3 mols, ncols=2 -> 2x2 = 4 axes: {n_axes}")
    matplotlib.pyplot.close(fig)

    print("\n✅  All tests completed!")

    # Opt-in Dash preview of the tooltip generation (blocking); run with `--dash`.
    if "--dash" in sys.argv:
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
