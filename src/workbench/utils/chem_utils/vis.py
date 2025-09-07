"""Molecular visualization utilities for Workbench"""

import logging
import base64
import re
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

# Set up the logger
log = logging.getLogger("workbench")


def _is_dark(color: str) -> bool:
    """Determine if an rgba color is dark based on RGB average.

    Args:
        color: Color in rgba(...) format

    Returns:
        True if the color is dark, False otherwise
    """
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color)
    if not match:
        log.warning(f"Invalid color format: {color}, defaulting to dark")
        return True  # Default to dark mode on error

    r, g, b = map(int, match.groups())
    return (r + g + b) / 3 < 128


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
    if _is_dark(background):
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

    # Encode SVG
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
        ("invalid_color", True, (0.25, 0.25, 0.25, 1.0)),  # Should use defaults
        ("rgba(255, 255, 255, 1)", False, (1.0, 1.0, 1.0, 1.0)),
        ("rgba(0, 0, 0, 1)", True, (0.0, 0.0, 0.0, 1.0)),
        ("rgba(64, 64, 64, 0.5)", True, (0.251, 0.251, 0.251, 0.5)),
        ("rgb(128, 128, 128)", False, (0.502, 0.502, 0.502, 1.0)),
    ]

    for color, expected_dark, expected_tuple in test_colors:
        is_dark_result = _is_dark(color)
        tuple_result = _rgba_to_tuple(color)

        dark_status = "✓" if is_dark_result == expected_dark else "✗"
        print(f"   {dark_status} is_dark('{color[:20]}...'): {is_dark_result} == {expected_dark}")

        # Check tuple values with tolerance for floating point
        tuple_match = all(abs(a - b) < 0.01 for a, b in zip(tuple_result, expected_tuple))
        tuple_status = "✓" if tuple_match else "✗"
        print(f"   {tuple_status} rgba_to_tuple('{color[:20]}...'): matches expected")

    # Test the show function (will open image windows)
    print("\n7. Testing show function (will open image windows)...")
    try:
        show(test_molecules["aspirin"])
        show(test_molecules["aspirin"], background="rgba(220, 220, 220, 1)")
        print("   ✓ show() function executed (check for image window)")
    except Exception as e:
        print(f"   ✗ show() function failed: {e}")

    print("\n✅ All tests completed!")
