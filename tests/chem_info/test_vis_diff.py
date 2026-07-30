"""Unit tests for molecule highlighting and MCS-based structural diffing."""

from workbench.utils.chem_utils.vis import (
    diff_molecules,
    img_from_smiles,
    stereo_differences,
    structural_differences,
    svg_from_smiles,
)

# The aqsol case that motivated this: same fingerprint, 9.7 log units apart.
CHROMIUM = "[Cr]"
CHROMIUM_TRIFLUORIDE = "[F].[F].[F].[Cr]"

# Bupivacaine, defined vs undefined stereocenter.
STEREO_DEFINED = "CCCCN1CCCC[C@H]1C(=O)NC1=C(C)C=CC=C1C"
STEREO_UNDEFINED = "CCCCN1CCCCC1C(=O)NC1=C(C)C=CC=C1C"

# Alanine enantiomers, and 2-butene geometry.
L_ALANINE = "C[C@H](N)C(=O)O"
D_ALANINE = "C[C@@H](N)C(=O)O"
TRANS_BUTENE = r"C/C=C/C"
CIS_BUTENE = r"C/C=C\C"

# Tartaric acid: two centers on a symmetric skeleton. The meso form written from either
# end is the same molecule with its R/S centers swapped in index order.
MESO_TARTARIC = "O[C@@H](C(=O)O)[C@H](O)C(=O)O"
MESO_TARTARIC_REVERSED = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
RR_TARTARIC = "O[C@@H](C(=O)O)[C@@H](O)C(=O)O"
SS_TARTARIC = "O[C@H](C(=O)O)[C@H](O)C(=O)O"


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


def test_stereo_only_pair_has_no_structural_difference():
    """MCS matches connectivity, so a stereo-only pair is identical to it."""
    assert structural_differences(STEREO_DEFINED, STEREO_UNDEFINED) == ([], [])


def test_enantiomers_differ_at_the_stereocenter():
    """Opposite R/S on the same skeleton flags the center atom and nothing else."""
    atoms, bonds = stereo_differences(L_ALANINE, D_ALANINE)

    assert atoms == [1]  # the alpha carbon
    assert bonds == []


def test_undefined_stereocenter_counts_as_a_difference():
    """Assigned versus undefined is a real difference, not a match."""
    atoms, _ = stereo_differences(STEREO_DEFINED, STEREO_UNDEFINED)

    assert len(atoms) == 1


def test_double_bond_geometry_is_reported_as_a_bond():
    """E versus Z flags the double bond rather than an atom."""
    atoms, bonds = stereo_differences(TRANS_BUTENE, CIS_BUTENE)

    assert atoms == []
    assert len(bonds) == 1


def test_symmetric_molecule_does_not_differ_from_itself():
    """A symmetric skeleton maps on several ways; the wrong one invents a difference."""
    assert stereo_differences(MESO_TARTARIC, MESO_TARTARIC_REVERSED) == ([], [])


def test_both_centers_flip_between_enantiomers():
    """(R,R) versus (S,S) differs at both centers, not one."""
    atoms, _ = stereo_differences(RR_TARTARIC, SS_TARTARIC)

    assert len(atoms) == 2


def test_diastereomers_differ_at_one_center():
    """(R,R) versus meso shares a center, so only the other one is flagged."""
    atoms, _ = stereo_differences(RR_TARTARIC, MESO_TARTARIC)

    assert len(atoms) == 1


def test_connectivity_change_is_not_a_stereo_difference():
    """The two comparisons stay independent — an amine swap carries no stereo."""
    assert stereo_differences("c1ccccc1CCO", "c1ccccc1CCN") == ([], [])


def test_stereo_differences_invalid_smiles_returns_none():
    """Matches structural_differences: None rather than raising."""
    assert stereo_differences("not_a_smiles", CHROMIUM) is None
    assert stereo_differences(CHROMIUM, "not_a_smiles") is None


def test_substituent_change_reports_atom_and_bond():
    """Swapping an alcohol for an amine flags the heteroatom and its bond."""
    atoms, bonds = structural_differences("c1ccccc1CCO", "c1ccccc1CCN")

    assert len(atoms) == 1
    assert len(bonds) == 1


def test_invalid_smiles_returns_none():
    """A bad SMILES is reported as None rather than raising."""
    assert structural_differences("not_a_smiles", CHROMIUM) is None
    assert structural_differences(CHROMIUM, "not_a_smiles") is None


def test_diff_molecules_returns_a_showable_figure():
    """Matches molecule_grid / neighborhood_graph so callers can just fig.show()."""
    fig = diff_molecules(CHROMIUM, CHROMIUM_TRIFLUORIDE, captions=["a", "b"])

    assert hasattr(fig, "savefig")
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["a", "b"]


def test_diff_molecules_renders_a_stereo_only_pair():
    """Enantiomers reach the renderer; what gets highlighted is covered by the diff tests."""
    fig = diff_molecules(L_ALANINE, D_ALANINE, captions=["L", "D"])

    assert len(fig.axes) == 2


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
