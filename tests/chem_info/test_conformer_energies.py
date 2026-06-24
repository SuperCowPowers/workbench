"""Unit tests for conformer energy ranking (MMFF vs GFN2-xTB).

These run locally (no AWS). The xTB-specific assertions skip when tblite is
not installed in the dev env; the dispatch/fallback assertions always run.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr

from workbench.utils.chem_utils.mol_descriptors_3d import (
    TBLITE_AVAILABLE,
    get_conformer_energies,
    xtb_conformer_energies,
    _forcefield_conformer_energies,
)

# Flexible, polar molecule — the regime where MMFF94s and GFN2-xTB rankings
# diverge most (near-zero rank correlation measured in the design spike).
DIPHENHYDRAMINE = "CN(C)CCOC(c1ccccc1)c1ccccc1"


def _optimized_mol(smiles: str, n_conf: int = 20) -> Chem.Mol:
    """Embed + MMFF94s-optimize a small conformer set (the pipeline's geometry)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=200, mmffVariant="MMFF94s")
    return mol


def test_dispatch_and_fallback_always_finite():
    """get_conformer_energies returns one finite energy per conformer either way."""
    mol = _optimized_mol(DIPHENHYDRAMINE)
    n = mol.GetNumConformers()

    xtb_default = get_conformer_energies(mol)  # method defaults to GFN2-xTB
    mmff_forced = get_conformer_energies(mol, method="MMFF")

    assert len(xtb_default) == n and len(mmff_forced) == n
    assert np.all(np.isfinite(mmff_forced))
    # Default path is finite whether xTB ran or fell back to MMFF.
    assert np.all(np.isfinite(xtb_default))


def test_mmff_forced_matches_forcefield_helper():
    """method='MMFF' bypasses xTB and equals the raw force-field energies."""
    mol = _optimized_mol(DIPHENHYDRAMINE)
    assert get_conformer_energies(mol, method="MMFF") == _forcefield_conformer_energies(mol)


@pytest.mark.skipif(not TBLITE_AVAILABLE, reason="tblite not installed in this env")
def test_xtb_ranking_diverges_from_mmff():
    """The whole point: GFN2-xTB re-ranks conformers vs MMFF94s.

    If the rankings were identical, reweighting would be pointless. On a
    flexible/polar molecule they should be weakly correlated at best.
    """
    mol = _optimized_mol(DIPHENHYDRAMINE)
    e_xtb = np.array(xtb_conformer_energies(mol))
    e_mmff = np.array(_forcefield_conformer_energies(mol))

    assert np.all(np.isfinite(e_xtb)), "xTB should score every conformer for this molecule"
    rho = spearmanr(e_mmff, e_xtb).correlation
    assert rho < 0.9, f"Expected MMFF/xTB ranking divergence, got spearman={rho:.3f}"


@pytest.mark.skipif(not TBLITE_AVAILABLE, reason="tblite not installed in this env")
def test_xtb_handles_formal_charge():
    """Charged species must not crash and must yield finite energies.

    Verifies the total-charge passthrough (a protonated tertiary amine).
    """
    mol = _optimized_mol("C[NH+](C)CCOC(c1ccccc1)c1ccccc1", n_conf=5)
    assert Chem.GetFormalCharge(mol) == 1
    e = np.array(xtb_conformer_energies(mol))
    assert np.all(np.isfinite(e))
