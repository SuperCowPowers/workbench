"""Unit tests for the UQ harness (regression and classification confidence).

These tests would have caught the `predict_intervals` overwrite bug — the key invariant
they enforce is that `prediction_std` is preserved through `predict_intervals`, and that
`compute_confidence` yields a ~uniform [0, 1] distribution when applied to the same data
used for calibration.

Run:
    pytest tests/confidence/test_uq_harness.py -v
"""
import numpy as np
import pandas as pd
import pytest

from workbench.model_script_utils.uq_harness import (
    apply_classification_confidence,
    calibrate_classification_confidence,
    calibrate_uq,
    compute_confidence,
    compute_vgmu_confidence,
    predict_intervals,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def regression_calibration_data():
    """Synthetic regression calibration data with heteroscedastic noise.

    Residuals are ~1.8× the ensemble std (realistic: ensemble std typically
    underestimates total error). This makes scale_factor_68 ≠ 1, so the
    round-trip test can detect a `prediction_std` overwrite.
    """
    rng = np.random.default_rng(42)
    n = 4000
    y_pred = rng.uniform(0, 10, size=n)
    # Heteroscedastic ensemble std: wide range from ~0.05 to ~2.0
    pred_std = np.abs(rng.normal(0, 0.5, size=n)) + 0.05
    # True y: noise scaled larger than pred_std → nonconformity > 1 → scale_68 > 1
    y_true = y_pred + rng.normal(0, 1.8, size=n) * pred_std
    return y_true, y_pred, pred_std


@pytest.fixture
def classification_calibration_data():
    """Synthetic 3-class ensemble probabilities with realistic disagreement."""
    rng = np.random.default_rng(42)
    n, n_classes, n_models = 2000, 3, 5

    # True labels
    y_true = rng.integers(0, n_classes, size=n)

    # Per-model probabilities: correct class gets higher prob on average
    all_probs = np.zeros((n_models, n, n_classes))
    for m in range(n_models):
        raw = rng.dirichlet([1.0] * n_classes, size=n)
        # Bias toward true class (noisily, so not all models agree)
        bias = rng.uniform(0.0, 0.7, size=n)
        for i in range(n):
            raw[i, y_true[i]] += bias[i]
            raw[i] /= raw[i].sum()
        all_probs[m] = raw

    avg_probs = all_probs.mean(axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    return avg_probs, all_probs, y_true, y_pred


# =============================================================================
# calibrate_uq tests
# =============================================================================
class TestCalibrateUQ:
    def test_returns_expected_keys(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        assert "confidence_levels" in uq
        assert "scale_factors" in uq
        assert "std_percentiles" in uq

    def test_std_percentiles_length(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        assert len(uq["std_percentiles"]) == 101  # 0th through 100th percentile

    def test_std_percentiles_monotonic(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        percentiles = np.asarray(uq["std_percentiles"])
        assert np.all(np.diff(percentiles) >= 0), "std_percentiles must be non-decreasing"

    def test_scale_factors_monotonic(self, regression_calibration_data):
        """Higher confidence levels must use larger scale factors."""
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        levels = sorted(uq["confidence_levels"])
        scales = [uq["scale_factors"][f"{lvl:.2f}"] for lvl in levels]
        assert all(scales[i] <= scales[i + 1] for i in range(len(scales) - 1))

    def test_coverage_matches_target(self, regression_calibration_data):
        """Conformal guarantee: on calibration data, coverage ≈ confidence level."""
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        safe_std = np.maximum(std, 1e-10)
        for lvl in uq["confidence_levels"]:
            q = uq["scale_factors"][f"{lvl:.2f}"]
            lower = y_pred - q * safe_std
            upper = y_pred + q * safe_std
            coverage = np.mean((y >= lower) & (y <= upper))
            # Conformal guarantees coverage >= level; allow small slack above
            assert coverage >= lvl - 0.01, (
                f"Coverage {coverage:.3f} below target {lvl:.2f}"
            )
            assert coverage <= lvl + 0.03, (
                f"Coverage {coverage:.3f} too far above target {lvl:.2f}"
            )


# =============================================================================
# predict_intervals tests (regression tests for the overwrite bug)
# =============================================================================
class TestPredictIntervals:
    def test_prediction_std_not_overwritten(self, regression_calibration_data):
        """Regression test: predict_intervals must NOT overwrite `prediction_std`.

        Pre-fix, predict_intervals replaced prediction_std with (q_84 - q_16)/2,
        which silently corrupted downstream compute_confidence.
        """
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std.copy()})
        original_std = df["prediction_std"].values.copy()

        df = predict_intervals(df, uq)
        np.testing.assert_allclose(
            df["prediction_std"].values,
            original_std,
            err_msg="predict_intervals must not overwrite prediction_std",
        )

    def test_emits_expected_quantile_columns(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = predict_intervals(df, uq)

        expected = ["q_025", "q_05", "q_10", "q_16", "q_25", "q_50", "q_75", "q_84", "q_90", "q_95", "q_975"]
        for col in expected:
            assert col in df.columns, f"Missing quantile column: {col}"

    def test_quantile_ordering(self, regression_calibration_data):
        """Quantile columns must be in ascending order for every row."""
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = predict_intervals(df, uq)

        q_cols = ["q_025", "q_05", "q_10", "q_16", "q_25", "q_50", "q_75", "q_84", "q_90", "q_95", "q_975"]
        for i in range(len(q_cols) - 1):
            lower = df[q_cols[i]].values
            upper = df[q_cols[i + 1]].values
            assert np.all(lower <= upper + 1e-10), (
                f"{q_cols[i]} should be <= {q_cols[i + 1]} for all rows"
            )

    def test_q50_equals_prediction(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = predict_intervals(df, uq)
        np.testing.assert_allclose(df["q_50"].values, y_pred)


# =============================================================================
# compute_confidence tests
# =============================================================================
class TestComputeConfidence:
    def test_confidence_range(self, regression_calibration_data):
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = compute_confidence(df, uq)
        assert df["confidence"].min() >= 0.0
        assert df["confidence"].max() <= 1.0

    def test_confidence_uniform_on_calibration_data(self, regression_calibration_data):
        """The core invariant: confidence on calibration data must be ~uniform on [0, 1].

        This is the test that would have caught the overwrite bug — pre-fix,
        confidence on calibration data was heavily skewed toward 0.
        """
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = compute_confidence(df, uq)

        q25, q50, q75 = np.percentile(df["confidence"].values, [25, 50, 75])
        # Uniform[0,1] would give exactly 0.25, 0.50, 0.75. Allow generous slack.
        assert 0.20 <= q25 <= 0.30, f"Expected q25≈0.25 for uniform confidence, got {q25:.3f}"
        assert 0.45 <= q50 <= 0.55, f"Expected q50≈0.50 for uniform confidence, got {q50:.3f}"
        assert 0.70 <= q75 <= 0.80, f"Expected q75≈0.75 for uniform confidence, got {q75:.3f}"

    def test_confidence_monotone_decreasing_in_std(self, regression_calibration_data):
        """Higher std must map to lower confidence (Spearman = -1 on calibration data)."""
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = compute_confidence(df, uq)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(df["prediction_std"].values, df["confidence"].values)
        assert rho < -0.99, f"Expected near-perfect inverse correlation, got {rho:.3f}"

    def test_pipeline_round_trip(self, regression_calibration_data):
        """Full pipeline: calibrate → predict_intervals → compute_confidence.

        Integration test ensuring predict_intervals doesn't corrupt what compute_confidence
        sees. If prediction_std is overwritten, confidence distribution will skew.
        """
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)

        df = pd.DataFrame({"prediction": y_pred, "prediction_std": std})
        df = predict_intervals(df, uq)
        df = compute_confidence(df, uq)

        q50 = np.median(df["confidence"].values)
        assert 0.45 <= q50 <= 0.55, (
            f"Pipeline round-trip: expected median confidence ≈ 0.5, got {q50:.3f}. "
            "Likely predict_intervals is mutating prediction_std."
        )

    def test_confidence_supports_custom_std_col(self, regression_calibration_data):
        """compute_confidence must honor the std_col parameter (used by multi-target)."""
        y, y_pred, std = regression_calibration_data
        uq = calibrate_uq(y, y_pred, std)
        df = pd.DataFrame({"prediction": y_pred, "target_a_pred_std": std})
        df = compute_confidence(df, uq, std_col="target_a_pred_std")

        assert "confidence" in df.columns
        q50 = np.median(df["confidence"].values)
        assert 0.45 <= q50 <= 0.55


# =============================================================================
# VGMU classification confidence tests
# =============================================================================
class TestClassificationConfidence:
    def test_vgmu_range(self, classification_calibration_data):
        avg_probs, all_probs, _, _ = classification_calibration_data
        raw = compute_vgmu_confidence(avg_probs, all_probs)
        assert raw.shape == (avg_probs.shape[0],)
        assert (raw >= 0).all() and (raw <= 1).all()

    def test_vgmu_zero_margin_gives_zero_confidence(self):
        """Uniform probs → margin = 0 → confidence = 0."""
        n, n_classes = 10, 3
        avg_probs = np.full((n, n_classes), 1.0 / n_classes)
        all_probs = np.broadcast_to(avg_probs, (5, n, n_classes)).copy()
        raw = compute_vgmu_confidence(avg_probs, all_probs)
        np.testing.assert_allclose(raw, 0.0, atol=1e-6)

    def test_vgmu_full_agreement_gracefully_degrades(self):
        """Single-model-like behavior (std=0) → gamma → 1 → confidence ≈ p_top1."""
        n, n_classes, n_models = 50, 3, 5
        rng = np.random.default_rng(0)
        base = rng.dirichlet([1.0] * n_classes, size=n)
        all_probs = np.broadcast_to(base, (n_models, n, n_classes)).copy()
        avg_probs = base.copy()
        raw = compute_vgmu_confidence(avg_probs, all_probs)
        p_top1 = np.max(avg_probs, axis=1)
        # With zero ensemble std, gamma = 1 - exp(-SNR) → 1 as SNR → inf
        # So raw should approach p_top1 from below
        np.testing.assert_allclose(raw, p_top1, atol=1e-6)

    def test_isotonic_calibration_round_trip(self, classification_calibration_data):
        """Calibrated confidence on the calibration data should track accuracy."""
        avg_probs, all_probs, y_true, y_pred = classification_calibration_data
        raw = compute_vgmu_confidence(avg_probs, all_probs)
        uq = calibrate_classification_confidence(raw, y_true, y_pred)
        calibrated = apply_classification_confidence(raw, uq["classification_confidence"])

        assert (calibrated >= 0).all() and (calibrated <= 1).all()

        # Calibrated confidence on the calibration data: high-conf bin accuracy > low-conf bin
        n_bins = 4
        bins = np.quantile(calibrated, np.linspace(0, 1, n_bins + 1))
        bins[-1] += 1e-9
        accs = []
        for i in range(n_bins):
            mask = (calibrated >= bins[i]) & (calibrated < bins[i + 1])
            if mask.sum() >= 50:
                accs.append((y_true[mask] == y_pred[mask]).mean())
        assert len(accs) >= 2
        assert accs[-1] > accs[0], (
            f"Calibrated confidence should discriminate accuracy across bins. "
            f"Bin accs: {accs}"
        )

    def test_isotonic_is_monotone(self, classification_calibration_data):
        """Isotonic calibration must preserve monotonicity: higher raw → higher calibrated."""
        avg_probs, all_probs, y_true, y_pred = classification_calibration_data
        raw = compute_vgmu_confidence(avg_probs, all_probs)
        uq = calibrate_classification_confidence(raw, y_true, y_pred)

        # Test on sorted raw values
        raw_sorted = np.sort(np.linspace(raw.min(), raw.max(), 100))
        calibrated = apply_classification_confidence(raw_sorted, uq["classification_confidence"])
        diffs = np.diff(calibrated)
        assert (diffs >= -1e-10).all(), "Calibration map must be monotone non-decreasing"


if __name__ == "__main__":
    # Allow running directly without pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
