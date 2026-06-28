"""Unit tests for the curated v2 3D descriptors (mol_descriptors_3d_v2).

Run locally (no AWS). Geometry/shape/surface assertions always run; the xTB
electronic-block assertions skip when tblite is not installed in the dev env
(the pipeline falls back to force-field energies + NaN electronics there).
"""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.chem_utils.mol_descriptors_3d import TBLITE_AVAILABLE
from workbench.utils.chem_utils.mol_descriptors_3d_v2 import (
    compute_descriptors_3d_v2,
    get_3d_v2_feature_names,
)

# Small, fast, chemically distinct set: polar, symmetric-apolar, rod, aromatic.
SMILES = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "decane": "CCCCCCCCCC",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


@pytest.fixture(scope="module")
def result() -> pd.DataFrame:
    df = pd.DataFrame({"smiles": list(SMILES.values()), "name": list(SMILES.keys())})
    return compute_descriptors_3d_v2(df).set_index("name")


def test_feature_count_is_26():
    assert len(get_3d_v2_feature_names()) == 26


def test_all_molecules_ok(result):
    assert (result["desc3d_status"] == "ok").all()


def test_all_features_present_and_finite(result):
    """Every declared v2 feature exists and is finite for these well-behaved molecules."""
    feats = get_3d_v2_feature_names()
    assert set(feats).issubset(result.columns)
    block = result[feats].astype(float)
    assert np.isfinite(block.to_numpy()).all(), block.isna().sum()[lambda s: s > 0].to_dict()


def test_surface_polar_split_is_sane(result):
    """Apolar hydrocarbons have ~zero polar SASA; aspirin (4 O) has substantial polar SASA."""
    assert result.loc["benzene", "surf_sasa_polar"] == pytest.approx(0.0, abs=1.0)
    assert result.loc["decane", "surf_sasa_polar"] == pytest.approx(0.0, abs=1.0)
    assert result.loc["aspirin", "surf_sasa_polar"] > 20.0
    # fraction apolar in [0, 1]
    assert (result["surf_frac_apolar"].between(0.0, 1.0)).all()


def test_shape_npr_classifies_rod_vs_disc(result):
    """NPR1 separates a linear chain (rod, low NPR1) from a flat aromatic (disc, ~0.5)."""
    assert result.loc["decane", "shape_npr1"] < result.loc["benzene", "shape_npr1"]
    assert result.loc["benzene", "shape_npr2"] == pytest.approx(0.5, abs=0.15)


def test_empty_and_invalid_are_skipped():
    """Bad input degrades gracefully to skip status + NaN features, no crash."""
    df = pd.DataFrame({"smiles": ["", "NOT_A_SMILES"]})
    out = compute_descriptors_3d_v2(df)
    assert list(out["desc3d_status"]) == ["skip:empty", "skip:parse"]
    assert out[get_3d_v2_feature_names()].isna().all().all()


def test_missing_smiles_column_raises():
    with pytest.raises(ValueError):
        compute_descriptors_3d_v2(pd.DataFrame({"mol": ["CCO"]}))


@pytest.mark.skipif(not TBLITE_AVAILABLE, reason="tblite not installed in this env")
def test_electronic_block_populated_with_xtb(result):
    """When xTB runs, the electronic features carry physically sane values."""
    assert (result["desc3d_energy_method"] == "GFN2-xTB").all()
    # Symmetric molecules have ~zero dipole; polar/asymmetric ones don't.
    assert result.loc["benzene", "elec_dipole"] == pytest.approx(0.0, abs=0.3)
    assert result.loc["decane", "elec_dipole"] == pytest.approx(0.0, abs=0.3)
    assert result.loc["ethanol", "elec_dipole"] > 1.0
    # Quadrupole is orthogonal to dipole: benzene has zero dipole but nonzero quadrupole.
    assert result.loc["benzene", "elec_quadrupole"] > 0.0
    # Frontier-orbital sanity: gap positive; saturated alkane wider than aromatic.
    assert (result["elec_gap"] > 0).all()
    assert result.loc["decane", "elec_gap"] > result.loc["benzene", "elec_gap"]


@pytest.mark.skipif(not TBLITE_AVAILABLE, reason="tblite not installed in this env")
def test_charge_weighted_psa_present_with_xtb(result):
    """surf_psa_charge (SASA weighted by |xTB charge|) is finite when charges are available."""
    assert np.isfinite(result["surf_psa_charge"].astype(float)).all()
    assert result.loc["aspirin", "surf_psa_charge"] > 0.0
