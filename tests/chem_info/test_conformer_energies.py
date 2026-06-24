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
    conformer_energies_and_method,
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


def test_energies_always_finite():
    """get_conformer_energies returns one finite energy per conformer.

    Finite whether xTB ran or fell back to the force field.
    """
    mol = _optimized_mol(DIPHENHYDRAMINE)
    energies = get_conformer_energies(mol)
    assert len(energies) == mol.GetNumConformers()
    assert np.all(np.isfinite(energies))


def test_forcefield_fallback_is_finite():
    """The MMFF/UFF fallback path (used when tblite is absent) is well-formed."""
    mol = _optimized_mol(DIPHENHYDRAMINE)
    ff_energies = _forcefield_conformer_energies(mol)
    assert len(ff_energies) == mol.GetNumConformers()
    assert np.all(np.isfinite(ff_energies))


def test_method_label_reports_actual_model():
    """The method label reflects what really produced the energies.

    This is what feeds desc3d_energy_method: GFN2-xTB when xTB ran, else the
    force-field name on fallback.
    """
    mol = _optimized_mol(DIPHENHYDRAMINE)
    _, method = conformer_energies_and_method(mol)
    assert method == ("GFN2-xTB" if TBLITE_AVAILABLE else "MMFF94s")


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
