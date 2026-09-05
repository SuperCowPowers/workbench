"""Coverage regression for V1's conformal calibration.

V1 sizes its conformal scale factors from out-of-fold error-model estimates. A
forest scoring the rows it trained on tracks them too closely, which compresses
the nonconformity spread and yields intervals narrower than nominal. These tests
pin that: calibrating in-sample undercovers on held-out queries, calibrating
out-of-fold lands on target.

Coverage is pooled across seeds — a single 100-row query set is too small for a
stable estimate at the 95% level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.uq_model_v1 import UQModelV1

from .fixtures import UQFixture, _fingerprints_to_strings

REGIME = "heteroskedastic"
SEEDS = range(8)

# (lower column, upper column, nominal coverage)
LEVELS = [("q_10", "q_90", 0.80), ("q_05", "q_95", 0.90), ("q_025", "q_975", 0.95)]


class _InSampleV1(UQModelV1):
    """V1 with the error model scoring the rows it was fit on."""

    @staticmethod
    def _oof_expected(error_model, X, y):
        return error_model.predict(X)


def _pooled_coverage(cls) -> dict[float, float]:
    """Fit `cls` per seed, score that seed's held-out queries, pool the hits."""
    hits = {nominal: 0 for _, _, nominal in LEVELS}
    total = 0

    for seed in SEEDS:
        fx = UQFixture.make(REGIME, seed=seed)
        prox = FingerprintProximity(fx.prox_df, id_column=fx.id_column, target=fx.target)

        uq = cls(prox, targets=[fx.target])
        uq.fit(fx.oof_ids, fx.y_true_oof, fx.y_pred_oof, fx.prediction_std_oof, target=fx.target)

        query = pd.DataFrame({fx.fingerprint_column: _fingerprints_to_strings(fx.query_fingerprints)})
        out = uq.predict(query, fx.y_pred_query, fx.prediction_std_query, target=fx.target)

        truth = np.asarray(fx.y_true_query, dtype=float)
        for lo_col, hi_col, nominal in LEVELS:
            lo = out[lo_col].to_numpy()
            hi = out[hi_col].to_numpy()
            hits[nominal] += int(np.sum((truth >= lo) & (truth <= hi)))
        total += len(truth)

    return {nominal: hits[nominal] / total for _, _, nominal in LEVELS}


@pytest.fixture(scope="module")
def coverage() -> dict[str, dict[float, float]]:
    """Pooled held-out coverage for both calibration strategies."""
    return {"oof": _pooled_coverage(UQModelV1), "in_sample": _pooled_coverage(_InSampleV1)}


def test_out_of_fold_calibration_hits_nominal_coverage(coverage):
    """Shipped calibration lands within 4 points of nominal on held-out queries."""
    for _, _, nominal in LEVELS:
        actual = coverage["oof"][nominal]
        assert abs(actual - nominal) <= 0.04, f"{nominal:.0%} interval covered {actual:.3f}"


def test_in_sample_calibration_undercovers(coverage):
    """The regression guard: scoring the fit rows shrinks intervals below nominal."""
    for _, _, nominal in LEVELS:
        assert coverage["in_sample"][nominal] < coverage["oof"][nominal], (
            f"at {nominal:.0%}, in-sample covered {coverage['in_sample'][nominal]:.3f} "
            f"vs out-of-fold {coverage['oof'][nominal]:.3f}"
        )

    # The gap should be substantial at the tails, not a rounding artifact
    assert coverage["oof"][0.95] - coverage["in_sample"][0.95] >= 0.02


def test_tiny_calibration_set_falls_back_to_in_sample():
    """Fewer than two rows can't be split, so calibration uses in-sample estimates."""
    from sklearn.ensemble import RandomForestRegressor

    X = np.array([[0.5, 0.1, 0.2, 0.3, 4.0, 0.05]])
    y = np.array([0.25])
    rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)

    out = UQModelV1._oof_expected(rf, X, y)

    assert out.shape == (1,)
    assert np.allclose(out, rf.predict(X))
