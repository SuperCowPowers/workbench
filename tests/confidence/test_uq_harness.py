"""Unit tests for the UQ harness (classification VGMU confidence).

Run:
    pytest tests/confidence/test_uq_harness.py -v
"""

import numpy as np
import pytest

from workbench.endpoints.uq_harness import (
    apply_classification_confidence,
    calibrate_classification_confidence,
    compute_vgmu_confidence,
)


# =============================================================================
# Fixtures
# =============================================================================
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
            f"Calibrated confidence should discriminate accuracy across bins. " f"Bin accs: {accs}"
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
