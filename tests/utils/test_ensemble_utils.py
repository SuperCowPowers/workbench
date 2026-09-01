"""Weight math shared by the deployed aggregation nodes and the offline simulator.

The invariant these protect: every weight row is finite. A non-finite weight becomes a
NaN prediction — served by a meta endpoint in one path, and in the other quietly dropped
by a NaN-skipping mean, which scores one strategy on a different row set than its rivals.
"""

import numpy as np
import pytest

from workbench.utils.ensemble_utils import conf_weights_with_fallback

FALLBACK = np.array([0.5, 0.3, 0.2])


def test_normal_confidences_normalize():
    conf = np.array([[1.0, 1.0, 2.0]])
    w = conf_weights_with_fallback(conf, FALLBACK)
    np.testing.assert_allclose(w, [[0.25, 0.25, 0.5]], atol=1e-9)
    assert np.isclose(w.sum(axis=1), 1.0)


def test_zero_confidence_row_uses_static_weights():
    w = conf_weights_with_fallback(np.zeros((1, 3)), FALLBACK)
    np.testing.assert_allclose(w, [FALLBACK])


def test_missing_confidence_uses_static_weights():
    # One member's confidence is absent — the row cannot be weighted by confidence.
    w = conf_weights_with_fallback(np.array([[0.9, np.nan, 0.4]]), FALLBACK)
    np.testing.assert_allclose(w, [FALLBACK])


def test_no_row_is_ever_non_finite():
    conf = np.array([[0.9, np.nan, 0.4], [0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [np.inf, 0.1, 0.1]])
    w = conf_weights_with_fallback(conf, FALLBACK)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_mixed_rows_fall_back_independently():
    conf = np.array([[1.0, 1.0, 2.0], [0.9, np.nan, 0.4]])
    w = conf_weights_with_fallback(conf, FALLBACK)
    np.testing.assert_allclose(w[0], [0.25, 0.25, 0.5], atol=1e-9)
    np.testing.assert_allclose(w[1], FALLBACK)


@pytest.mark.parametrize("missing_frac", [0.1, 0.9])
def test_predictions_stay_finite_for_every_row(missing_frac):
    """The CYP case: most rows have a member with no confidence, and every row still scores."""
    rng = np.random.default_rng(0)
    n = 200
    conf = rng.uniform(0.1, 0.9, size=(n, 3))
    conf[rng.random(n) < missing_frac, 1] = np.nan
    preds = rng.normal(5.0, 1.0, size=(n, 3))
    weights = conf_weights_with_fallback(conf, FALLBACK)
    combined = (preds * weights).sum(axis=1)
    assert np.isfinite(combined).all(), "every row must produce a prediction"
