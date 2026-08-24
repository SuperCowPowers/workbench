"""Tests for the soft-threshold (credible-interval aware) regression metrics."""

import numpy as np
import pytest

from workbench.utils.metrics_utils import soft_threshold_error, soft_threshold_rae


class TestSoftThresholdError:
    def test_inside_interval_is_zero(self):
        err = soft_threshold_error(y_pred=[5.0, 5.5, 6.0], ci_lower=[4.9, 4.9, 4.9], ci_upper=[6.1, 6.1, 6.1])
        assert np.allclose(err, 0.0)

    def test_on_the_bound_is_zero(self):
        err = soft_threshold_error(y_pred=[4.9, 6.1], ci_lower=[4.9, 4.9], ci_upper=[6.1, 6.1])
        assert np.allclose(err, 0.0)

    def test_distance_to_nearest_bound(self):
        err = soft_threshold_error(y_pred=[4.0, 7.0], ci_lower=[4.9, 4.9], ci_upper=[6.1, 6.1])
        assert np.allclose(err, [0.9, 0.9])

    def test_zero_width_interval_matches_absolute_error(self):
        y_true, y_pred = np.array([5.0, 6.0]), np.array([5.5, 4.0])
        err = soft_threshold_error(y_pred, ci_lower=y_true, ci_upper=y_true)
        assert np.allclose(err, np.abs(y_true - y_pred))


class TestSoftThresholdRAE:
    def test_perfect_predictions_score_zero(self):
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        score = soft_threshold_rae(y_true, y_true, y_true - 0.1, y_true + 0.1)
        assert score == pytest.approx(0.0)

    def test_predictions_inside_intervals_score_zero(self):
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        score = soft_threshold_rae(y_true, y_true + 0.05, y_true - 0.1, y_true + 0.1)
        assert score == pytest.approx(0.0)

    def test_soft_baseline_makes_the_mean_predictor_score_one(self):
        """What soft_baseline=True buys: 1.0 means 'no better than the mean'."""
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        mean_pred = np.full_like(y_true, y_true.mean())
        score = soft_threshold_rae(y_true, mean_pred, y_true - 0.1, y_true + 0.1, soft_baseline=True)
        assert score == pytest.approx(1.0)

    def test_default_matches_the_published_denominator(self):
        """The default reproduces OpenADMET's rae(): a plain sum|y - mean(y)| denominator."""
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        y_pred = y_true + 0.5
        ci_lower, ci_upper = y_true - 0.1, y_true + 0.1

        score = soft_threshold_rae(y_true, y_pred, ci_lower, ci_upper)
        expected = soft_threshold_error(y_pred, ci_lower, ci_upper).sum() / np.abs(y_true - y_true.mean()).sum()
        assert score == pytest.approx(expected)

    def test_soft_baseline_inflates_the_score(self):
        """The two conventions are not interconvertible; True always reads higher."""
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        y_pred = y_true + 0.5
        published = soft_threshold_rae(y_true, y_pred, y_true - 0.1, y_true + 0.1)
        soft = soft_threshold_rae(y_true, y_pred, y_true - 0.1, y_true + 0.1, soft_baseline=True)
        assert soft > published

    def test_beating_the_mean_scores_below_one(self):
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        score = soft_threshold_rae(y_true, y_true + 0.3, y_true - 0.1, y_true + 0.1)
        assert 0.0 < score < 1.0

    def test_wider_intervals_never_increase_the_score(self):
        """Widening credible intervals forgives error, so the score must not rise."""
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        y_pred = y_true + 0.5
        tight = soft_threshold_rae(y_true, y_pred, y_true - 0.1, y_true + 0.1)
        wide = soft_threshold_rae(y_true, y_pred, y_true - 0.4, y_true + 0.4)
        assert wide <= tight

    def test_plain_baseline_deflates_the_score(self):
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        y_pred = y_true + 0.5
        soft = soft_threshold_rae(y_true, y_pred, y_true - 0.1, y_true + 0.1, soft_baseline=True)
        plain = soft_threshold_rae(y_true, y_pred, y_true - 0.1, y_true + 0.1, soft_baseline=False)
        assert plain < soft

    def test_zero_baseline_error_returns_nan(self):
        """A constant target has no spread for the mean predictor to miss."""
        y_true = np.array([5.0, 5.0, 5.0])
        score = soft_threshold_rae(y_true, y_true + 1.0, y_true - 0.1, y_true + 0.1)
        assert np.isnan(score)
