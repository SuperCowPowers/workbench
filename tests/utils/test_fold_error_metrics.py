"""Tests for the fold-error metrics in metrics_utils."""

import numpy as np
import pytest

from workbench.utils.metrics_utils import afe, gmfe, within_fold

TRUE = np.array([1.0, 10.0, 100.0, 4.0])
PRED = np.array([2.0, 5.0, 100.0, 16.0])  # 2x high, 2x low, exact, 4x high


def test_gmfe_is_the_geometric_mean_of_the_fold_misses():
    assert gmfe(TRUE, PRED) == pytest.approx((2 * 2 * 1 * 4) ** 0.25)
    assert gmfe(TRUE, TRUE) == pytest.approx(1.0)


def test_afe_keeps_the_sign_that_gmfe_drops():
    assert afe(TRUE, PRED) == pytest.approx((2 * 0.5 * 1 * 4) ** 0.25)
    assert afe(TRUE, TRUE * 3) == pytest.approx(3.0)
    assert afe(TRUE, TRUE / 3) == pytest.approx(1 / 3)


def test_within_fold_counts_both_directions_and_the_boundary():
    assert within_fold(TRUE, PRED) == pytest.approx(0.75)
    assert within_fold(TRUE, PRED, fold=4.0) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [gmfe, afe, within_fold])
def test_nonpositive_values_raise(metric):
    with pytest.raises(ValueError):
        metric(TRUE, np.array([2.0, 0.0, 100.0, 16.0]))
