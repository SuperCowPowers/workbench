"""Unit tests for regression-UQ proximity backend selection and query slicing.

Covers the two decisions the templates delegate to ``uq_regression``:

    1. ``_build_proximity`` — which Proximity backend gets built from a given
       reference set (smiles wins, else the model's features, else nothing).
    2. ``uq_query_df`` — what payload a fit UQ model's backend needs at
       inference, so callers never sniff columns themselves.

Run:
    pytest tests/uq/test_backend_selection.py -v
"""

import numpy as np
import pandas as pd
import pytest

from workbench.endpoints.uq_regression import _build_proximity, fit_regression_uq, uq_query_df

FEATURES = ["f1", "f2", "f3"]
SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCC", "CCOC", "c1ccncc1", "CC(C)O", "CCCCO"]


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def feature_ref_df():
    """Reference set with numeric features and no smiles column."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["id"] = [f"r{i}" for i in range(n)]
    df["y"] = 2 * df.f1 - df.f2 + rng.normal(scale=0.3, size=n)
    df["in_model"] = True
    return df


@pytest.fixture
def smiles_ref_df():
    """Reference set with a smiles column (structure-based neighborhoods)."""
    rng = np.random.default_rng(0)
    smiles = SMILES * 8
    df = pd.DataFrame({"id": [f"m{i}" for i in range(len(smiles))], "smiles": smiles})
    df["y"] = rng.normal(size=len(smiles))
    df["in_model"] = True
    return df


def _fit(ref_df, features, active_version="v1"):
    """Fit all three UQ versions on a reference set, returning the fit_regression_uq dict."""
    rng = np.random.default_rng(1)
    n = len(ref_df)
    y_true = ref_df["y"].values
    y_pred = y_true + rng.normal(scale=0.3, size=n)
    y_std = np.abs(rng.normal(0.3, 0.1, n))
    return fit_regression_uq(
        per_target={"y": {"ids": ref_df["id"].tolist(), "y_true": y_true, "y_pred": y_pred, "y_std": y_std}},
        prox_df=ref_df,
        id_column="id",
        features=features,
        active_version=active_version,
    )


# =============================================================================
# _build_proximity — backend selection
# =============================================================================
def test_smiles_builds_fingerprint_backend(smiles_ref_df):
    prox = _build_proximity(smiles_ref_df, id_column="id", targets=["y"], features=None)
    assert prox.space == "fingerprint"


def test_smiles_wins_over_features(smiles_ref_df):
    """A reference set carrying both still gets structure-based neighborhoods."""
    df = smiles_ref_df.copy()
    for f in FEATURES:
        df[f] = np.arange(len(df), dtype=float)
    prox = _build_proximity(df, id_column="id", targets=["y"], features=FEATURES)
    assert prox.space == "fingerprint"


def test_features_build_feature_space_backend(feature_ref_df):
    prox = _build_proximity(feature_ref_df, id_column="id", targets=["y"], features=FEATURES)
    assert prox.space == "features"
    assert prox.features == FEATURES


def test_no_smiles_and_no_features_builds_nothing(feature_ref_df):
    assert _build_proximity(feature_ref_df, id_column="id", targets=["y"], features=None) is None
    assert _build_proximity(None, id_column="id", targets=["y"], features=FEATURES) is None


def test_features_absent_from_reference_set_builds_nothing(feature_ref_df):
    """Named features that aren't actually columns can't index anything."""
    prox = _build_proximity(feature_ref_df, id_column="id", targets=["y"], features=["nope", "also_nope"])
    assert prox is None


# =============================================================================
# fit_regression_uq — v1/v2 fit on both backends, v0 fallback when neither
# =============================================================================
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_fits_on_feature_space(feature_ref_df, version):
    uq = _fit(feature_ref_df, FEATURES, active_version=version)
    assert uq["uq_model"].UQ_VERSION == version
    assert uq["uq_model"].prox.space == "features"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_fits_on_fingerprints(smiles_ref_df, version):
    uq = _fit(smiles_ref_df, None, active_version=version)
    assert uq["uq_model"].UQ_VERSION == version
    assert uq["uq_model"].prox.space == "fingerprint"


def test_falls_back_to_v0_without_any_neighborhood(feature_ref_df):
    uq = _fit(feature_ref_df, features=None, active_version="v1")
    assert uq["uq_model"].UQ_VERSION == "v0"
    assert uq["v1"] is None and uq["v2"] is None
    assert uq["v0"] is not None


# =============================================================================
# uq_query_df — what each backend wants at inference
# =============================================================================
def test_query_df_is_none_for_v0(feature_ref_df):
    """V0 has no backend; None is the expected payload, not a skip signal."""
    uq = _fit(feature_ref_df, features=None, active_version="v1")
    assert uq_query_df(uq["uq_model"], feature_ref_df) is None


def test_query_df_slices_smiles_for_fingerprints(smiles_ref_df):
    uq = _fit(smiles_ref_df, None)
    batch = pd.DataFrame({"smiles": ["CCOCCO", "c1ccc(F)cc1"], "unrelated": [1, 2]})
    assert list(uq_query_df(uq["uq_model"], batch).columns) == ["smiles"]


def test_query_df_slices_features_for_feature_space(feature_ref_df):
    uq = _fit(feature_ref_df, FEATURES)
    rng = np.random.default_rng(2)
    batch = pd.DataFrame({f: rng.normal(size=4) for f in FEATURES})
    batch["unrelated"] = "x"
    assert list(uq_query_df(uq["uq_model"], batch).columns) == FEATURES


def test_query_df_is_none_when_a_feature_is_missing(feature_ref_df):
    """Feature space is all-or-nothing — the scaler needs every column it was fit on."""
    uq = _fit(feature_ref_df, FEATURES)
    rng = np.random.default_rng(3)
    batch = pd.DataFrame({f: rng.normal(size=4) for f in FEATURES[:-1]})
    assert uq_query_df(uq["uq_model"], batch) is None


def test_feature_space_predict_produces_confidence(feature_ref_df):
    """The full template path: fit -> slice a novel batch -> score it."""
    uq = _fit(feature_ref_df, FEATURES)
    rng = np.random.default_rng(4)
    batch = pd.DataFrame({f: rng.normal(size=6) for f in FEATURES})
    out = uq["uq_model"].predict(uq_query_df(uq["uq_model"], batch), np.zeros(6), np.full(6, 0.3))
    assert "confidence" in out.columns
    assert not out["confidence"].isna().any()
