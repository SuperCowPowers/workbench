"""Unit tests for molecule highlighting and MCS-based structural diffing."""

from workbench.utils.chem_utils.vis import (
    diff_molecules,
    img_from_smiles,
    structural_differences,
    svg_from_smiles,
)

# The aqsol case that motivated this: same fingerprint, 9.7 log units apart.
CHROMIUM = "[Cr]"
CHROMIUM_TRIFLUORIDE = "[F].[F].[F].[Cr]"

# Bupivacaine, defined vs undefined stereocenter.
STEREO_DEFINED = "CCCCN1CCCC[C@H]1C(=O)NC1=C(C)C=CC=C1C"
STEREO_UNDEFINED = "CCCCN1CCCCC1C(=O)NC1=C(C)C=CC=C1C"


def test_extra_fragments_are_reported_as_differences():
    """The trifluoride's three fluorines are what it has beyond the shared core."""
    atoms, _ = structural_differences(CHROMIUM_TRIFLUORIDE, CHROMIUM)

    assert len(atoms) == 3


def test_subset_molecule_has_no_differences():
    """Bare chromium is entirely contained in the trifluoride, so nothing differs."""
    atoms, bonds = structural_differences(CHROMIUM, CHROMIUM_TRIFLUORIDE)

    assert atoms == []
    assert bonds == []


def test_counterions_are_the_difference_between_two_salts():
    """Two sulfates differ only by their cations — the sulfate core is shared."""
    atoms, _ = structural_differences("[Na+].[Na+].[O-][S]([O-])(=O)=O", "[Ca++].[O-][S]([O-])(=O)=O")

    assert len(atoms) == 2  # the two sodiums


def test_stereo_only_pair_highlights_nothing():
    """MCS matches connectivity, so a stereo-only difference is invisible here.

    This is documented behavior, and the empty result is itself diagnostic: a
    coincident pair with no structural diff differs only in stereo or geometry.
    """
    assert structural_differences(STEREO_DEFINED, STEREO_UNDEFINED) == ([], [])


def test_substituent_change_reports_atom_and_bond():
    """Swapping an alcohol for an amine flags the heteroatom and its bond."""
    atoms, bonds = structural_differences("c1ccccc1CCO", "c1ccccc1CCN")

    assert len(atoms) == 1
    assert len(bonds) == 1


def test_invalid_smiles_returns_none():
    """A bad SMILES is reported as None rather than raising."""
    assert structural_differences("not_a_smiles", CHROMIUM) is None
    assert structural_differences(CHROMIUM, "not_a_smiles") is None


def test_diff_molecules_renders_two_panels():
    """The side-by-side carries one SVG per molecule."""
    svg = diff_molecules(CHROMIUM, CHROMIUM_TRIFLUORIDE, captions=["a", "b"])

    assert svg.count("<svg") == 2


def test_diff_molecules_none_on_invalid_input():
    """An unparseable member means no rendering."""
    assert diff_molecules("not_a_smiles", CHROMIUM) is None


def test_svg_highlight_returns_raw_markup_when_not_encoded():
    """encode=False yields SVG markup rather than a data URI."""
    svg = svg_from_smiles("c1ccccc1CCO", highlight_atoms=[6, 7, 8], encode=False)

    assert svg.lstrip().startswith("<?xml")
    assert "<svg" in svg


def test_svg_highlight_encodes_by_default():
    """The default stays a base64 data URI, as callers expect."""
    uri = svg_from_smiles("c1ccccc1CCO", highlight_atoms=[6])

    assert uri.startswith("data:image/svg+xml;base64,")


def test_img_accepts_highlights():
    """Highlighting is a passthrough on the PIL path too."""
    assert img_from_smiles("c1ccccc1CCO", highlight_atoms=[6, 7, 8]) is not None


def test_img_invalid_smiles_returns_none():
    """Unchanged behavior for bad input."""
    assert img_from_smiles("not_a_smiles") is None
