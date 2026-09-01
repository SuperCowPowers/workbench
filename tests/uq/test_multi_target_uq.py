"""Multi-target regression UQ: per-target calibration over one shared proximity.

A multi-target model's reference set is sparse — most rows carry a label for only
some of the targets. These tests pin the three behaviors that makes workable:

    1. Each target is calibrated on its own labels, so `{target}_confidence` is
       that target's confidence and not the primary's under another name.
    2. Target statistics come from the nearest *labeled* neighbors, so a sparse
       target still gets confidence rather than a mostly-NaN column.
    3. Standard columns with no per-target counterpart are dropped from a
       per-target capture rather than left holding the primary target's values.

Run:
    pytest tests/uq/test_multi_target_uq.py -v
"""

import numpy as np
import pandas as pd
import pytest

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.residual_features import ResidualFeatures
from workbench.algorithms.dataframe.uq_model_v0 import UQModelV0
from workbench.algorithms.dataframe.uq_model_v1 import UQModelV1
from workbench.algorithms.dataframe.uq_model_v2 import UQModelV2
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.endpoints.uq_regression import fit_regression_uq

DENSE = "dense_target"
SPARSE = "sparse_target"

# Enough distinct scaffolds that neighborhoods aren't degenerate
_SMILES_POOL = [
    "CCO",
    "CCN",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCOC",
    "CCOCC",
    "CC(C)O",
    "CC(C)C",
    "CC(=O)O",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "c1ccncc1",
    "Cc1ccncc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "CC(N)=O",
    "CCC(=O)O",
    "CCCO",
    "CCCN",
    "COC",
    "CSC",
    "CCS",
    "CC#N",
]


@pytest.fixture
def multi_target_df():
    """Reference set where the sparse target is labeled on only ~15% of rows."""
    rng = np.random.default_rng(0)
    smiles = _SMILES_POOL * 8
    n = len(smiles)
    df = pd.DataFrame({"id": [f"m{i}" for i in range(n)], "smiles": smiles})
    df[DENSE] = rng.normal(size=n)
    df[SPARSE] = rng.normal(loc=5.0, scale=2.0, size=n)
    # Blank out most of the sparse target — the multi-target reality
    blank = rng.permutation(n)[: int(n * 0.85)]
    df.loc[blank, SPARSE] = np.nan
    df["in_model"] = True
    return df


def _per_target(df, target, rng):
    """The out-of-fold payload fit_regression_uq wants for one target."""
    labeled = df[df[target].notna()]
    y_true = labeled[target].values
    y_pred = y_true + rng.normal(scale=0.3, size=len(labeled))
    return {
        "ids": labeled["id"].tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_std": np.abs(rng.normal(0.3, 0.1, len(labeled))),
    }


@pytest.fixture
def fit_dict(multi_target_df):
    rng = np.random.default_rng(1)
    return fit_regression_uq(
        per_target={
            DENSE: _per_target(multi_target_df, DENSE, rng),
            SPARSE: _per_target(multi_target_df, SPARSE, rng),
        },
        prox_df=multi_target_df,
        id_column="id",
        active_version="v1",
    )


# =============================================================================
# ResidualFeatures — per-target aggregation off one shared index
# =============================================================================
def test_residual_features_targets_the_column_it_is_given(multi_target_df):
    """One proximity, two targets, two different neighborhood summaries."""
    prox = FingerprintProximity(multi_target_df, id_column="id", target=[DENSE, SPARSE])
    ids = multi_target_df["id"].tolist()

    dense = ResidualFeatures(prox, target=DENSE).compute(ids, k=10)
    sparse = ResidualFeatures(prox, target=SPARSE).compute(ids, k=10)

    # The sparse target sits around 5.0, the dense one around 0.0
    assert dense["knn_target_mean"].mean() < 2.0
    assert sparse["knn_target_mean"].mean() > 3.0
    # Distance is target-independent — same index, same neighbors
    pd.testing.assert_series_equal(dense["knn_distance"], sparse["knn_distance"])


def test_sparse_target_still_gets_labeled_neighbors(multi_target_df):
    """Over-fetching by label coverage is what keeps confidence from going NaN."""
    prox = FingerprintProximity(multi_target_df, id_column="id", target=[DENSE, SPARSE])
    feat = ResidualFeatures(prox, target=SPARSE).compute(multi_target_df["id"].tolist(), k=10)

    # Every query resolved, and nearly all found at least two labeled neighbors
    assert feat["knn_distance"].notna().all()
    assert (feat["knn_target_count"] >= 2).mean() > 0.9
    assert feat["knn_target_std"].notna().mean() > 0.9


def test_unlabeled_neighborhood_reports_zero_count_not_nan(multi_target_df):
    """A target with no labels at all is a count of zero, not a missing count."""
    df = multi_target_df.copy()
    df["empty_target"] = np.nan
    prox = FingerprintProximity(df, id_column="id", target=[DENSE, "empty_target"])
    feat = ResidualFeatures(prox, target="empty_target").compute(df["id"].tolist(), k=10)

    assert (feat["knn_target_count"] == 0).all()
    assert feat["knn_distance"].notna().all()


def test_residual_features_rejects_unknown_target(multi_target_df):
    prox = FingerprintProximity(multi_target_df, id_column="id", target=DENSE)
    with pytest.raises(ValueError, match="not a column"):
        ResidualFeatures(prox, target="nope")


# =============================================================================
# Per-target calibration — each target gets its own confidence
# =============================================================================
@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
def test_every_target_is_calibrated(fit_dict, version):
    """No target is left reading the primary's calibration."""
    uq = fit_dict[version]
    fitted = (
        uq.error_models if version == "v1" else (uq.residual_calibrator if version == "v0" else uq.variance_percentiles)
    )
    assert set(fitted) == {DENSE, SPARSE}
    assert uq.primary_target == DENSE


def test_per_target_confidence_differs(fit_dict, multi_target_df):
    """The whole point: naming a target changes the confidence you get back."""
    uq = fit_dict["v1"]
    query = multi_target_df[["smiles"]]
    preds = np.zeros(len(query))
    stds = np.full(len(query), 0.3)

    dense = uq.predict(query, preds, stds, target=DENSE)["confidence"]
    sparse = uq.predict(query, preds, stds, target=SPARSE)["confidence"]

    assert not np.allclose(dense.values, sparse.values)
    # And the default is the primary, not something else
    default = uq.predict(query, preds, stds)["confidence"]
    np.testing.assert_allclose(default.values, dense.values)


def test_sparse_target_confidence_is_populated(fit_dict, multi_target_df):
    """The bug this fixes: a sparse target's confidence was mostly NaN."""
    uq = fit_dict["v1"]
    labeled = multi_target_df[multi_target_df[SPARSE].notna()]
    out = uq.predict(labeled[["smiles"]], labeled[SPARSE].values, np.full(len(labeled), 0.3), target=SPARSE)
    assert out["confidence"].notna().all()


def test_predict_rejects_uncalibrated_target(fit_dict, multi_target_df):
    uq = fit_dict["v1"]
    with pytest.raises(RuntimeError, match="no fitted error model"):
        uq.predict(
            multi_target_df[["smiles"]], np.zeros(len(multi_target_df)), np.ones(len(multi_target_df)), target="nope"
        )


# =============================================================================
# Persistence — per-target state survives a round trip
# =============================================================================
@pytest.mark.parametrize("version,loader", [("v0", UQModelV0), ("v1", UQModelV1), ("v2", UQModelV2)])
def test_save_load_round_trip(fit_dict, multi_target_df, tmp_path, version, loader):
    model_dir = str(tmp_path)
    fit_dict["v1"].save(model_dir)  # writes the shared uq_proximity.joblib
    fit_dict[version].save(model_dir)

    reloaded = loader.load(model_dir)
    assert reloaded.targets == [DENSE, SPARSE]

    query = multi_target_df[["smiles"]]
    preds, stds = np.zeros(len(query)), np.full(len(query), 0.3)
    for target in (DENSE, SPARSE):
        before = fit_dict[version].predict(query, preds, stds, target=target)["confidence"]
        after = reloaded.predict(query, preds, stds, target=target)["confidence"]
        np.testing.assert_allclose(before.values, after.values, equal_nan=True)


def test_saved_proximity_keeps_every_target(fit_dict, tmp_path):
    """The slim proximity has to carry all targets, or reload loses the sparse one."""
    model_dir = str(tmp_path)
    fit_dict["v1"].save(model_dir)
    slim = UQModelV1.load(model_dir).prox
    assert DENSE in slim.df.columns
    assert SPARSE in slim.df.columns


# =============================================================================
# Capture remap — no cross-target leakage
# =============================================================================
def test_remap_points_standard_columns_at_the_target():
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "prediction": [1.0, 2.0],
            "confidence": [0.1, 0.2],
            f"{DENSE}_pred": [1.0, 2.0],
            f"{DENSE}_confidence": [0.1, 0.2],
            f"{SPARSE}_pred": [7.0, 8.0],
            f"{SPARSE}_confidence": [0.9, 0.8],
        }
    )
    remapped = EndpointCore._remap_multi_target_columns(df, SPARSE)
    assert remapped["prediction"].tolist() == [7.0, 8.0]
    assert remapped["confidence"].tolist() == [0.9, 0.8]


def test_remap_drops_rather_than_leaks_the_primary():
    """A target with no confidence of its own must not inherit the primary's."""
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "prediction": [1.0, 2.0],
            "confidence": [0.1, 0.2],
            f"{SPARSE}_pred": [7.0, 8.0],
        }
    )
    remapped = EndpointCore._remap_multi_target_columns(df, SPARSE)
    assert remapped["prediction"].tolist() == [7.0, 8.0]
    assert "confidence" not in remapped.columns
