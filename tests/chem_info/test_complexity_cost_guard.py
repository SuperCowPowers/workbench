"""Unit tests for the xTB cost backstop in check_complexity (skip:cost).

Local + fast (no AWS). Verifies that large + very flexible molecules — which
pass the size/topology guards but would time out the GFN2-xTB energy step — are
caught with a clean ``skip:cost`` status, while normal and merely-large-OR-
merely-flexible molecules still pass.
"""

from rdkit import Chem

from workbench.utils.chem_utils.mol_descriptors_3d import (
    check_complexity,
    adaptive_n_conformers,
    MAX_CONFORMER_ATOM_COST,
)

# Irganox 1010 (CAS 6683-19-8): public, IP-free large/flexible surrogate for the
# proprietary dye that motivated this guard. 85 heavy × 500 conformers = 42500.
IRGANOX_1010 = (
    "O=C(OCC(COC(=O)CCc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)(COC(=O)CCc1cc(C(C)(C)C)"
    "c(O)c(C(C)(C)C)c1)COC(=O)CCc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)CCc1cc(C(C)(C)C)"
    "c(O)c(C(C)(C)C)c1"
)


def _check_full(smiles: str):
    """Run check_complexity with the full-mode (adaptive) conformer count."""
    mol = Chem.MolFromSmiles(smiles)
    return check_complexity(mol, n_conformers=adaptive_n_conformers(mol))


def test_irganox_triggers_skip_cost():
    """The surrogate is caught by the cost backstop in full mode."""
    assert _check_full(IRGANOX_1010) == "skip:cost"


def test_irganox_passes_size_guards():
    """It's not caught by the size/topology guards — that's the whole point.

    Without a conformer count (size/topology guards only), it passes; the
    skip is specifically the cost backstop.
    """
    mol = Chem.MolFromSmiles(IRGANOX_1010)
    assert mol.GetNumHeavyAtoms() <= 150  # under the heavy-atom guard
    assert check_complexity(mol) is None  # no n_conformers => no cost check


def test_fast_mode_never_skips_on_cost():
    """Fast mode (10 conformers) is cheap, so cost never trips even for Irganox."""
    mol = Chem.MolFromSmiles(IRGANOX_1010)
    assert check_complexity(mol, n_conformers=10) is None


def test_flexible_but_small_passes():
    """High flexibility alone is fine — docosane (C22, tier 500) stays OK.

    Guards against an over-aggressive threshold that would break the existing
    flexibility reference set.
    """
    assert _check_full("CCCCCCCCCCCCCCCCCCCCCC") is None  # docosane: 22 heavy × 500 = 11k


def test_large_but_rigid_passes():
    """Large but rigid (low rotatable bonds => tier 50) stays well under budget."""
    # A bis-azo disulfonate dye (Congo-Red-like): ~46 heavy, ~7 rot => tier 50.
    congo_like = "Nc1ccc2cc(S(=O)(=O)O)c(/N=N/c3ccc(-c4ccc(/N=N/c5ccc6ccc(S(=O)(=O)O)cc6c5N)cc4)cc3)cc2c1"
    assert _check_full(congo_like) is None


def test_threshold_is_the_boundary():
    """cost == MAX passes; cost > MAX skips (boundary is exclusive)."""
    mol = Chem.MolFromSmiles(IRGANOX_1010)
    n_heavy = mol.GetNumHeavyAtoms()
    at_limit = MAX_CONFORMER_ATOM_COST // n_heavy           # heavy * this <= MAX
    over_limit = at_limit + 1
    assert check_complexity(mol, n_conformers=at_limit) is None
    assert check_complexity(mol, n_conformers=over_limit) == "skip:cost"
