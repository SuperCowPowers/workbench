"""Tests for placement_metrics: where predictions sit relative to the targets."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.metrics_utils import placement_metrics


def _std(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def _frame(rho: float, k: float, b: float, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Predictions carrying an exact correlation, spread ratio and offset.

    The targets are standardized, so `b` is already in sd(true) units and the frame's
    spread_ratio comes back as exactly `k`.
    """
    rng = np.random.default_rng(seed)
    y = _std(rng.normal(0, 1, n))
    noise = rng.normal(0, 1, n)
    # Orthogonalize against y so the realized correlation is rho, not rho plus sampling drift
    noise = _std(noise - (noise @ y) / (y @ y) * y)
    pred = k * (rho * y + np.sqrt(1 - rho**2) * noise) + b
    return pd.DataFrame({"y": y, "prediction": pred})


class TestIdentities:
    def test_gap_is_r2_opt_minus_r2(self):
        m = placement_metrics(_frame(rho=0.8, k=0.6, b=0.3), "y")
        assert m["gap"] == pytest.approx(m["r2_opt"] - m["r2"])

    def test_r2_opt_is_squared_correlation(self):
        m = placement_metrics(_frame(rho=0.8, k=0.6, b=0.3), "y")
        assert m["r2_opt"] == pytest.approx(0.8**2)

    def test_gap_splits_into_spread_and_offset_terms(self):
        """gap == (pearsonr - spread_ratio)**2 + (bias/sd(true))**2, exactly."""
        rho, k, b = 0.8, 0.6, 0.3
        m = placement_metrics(_frame(rho, k, b), "y")
        assert m["gap"] == pytest.approx((rho - m["spread_ratio"]) ** 2 + m["bias"] ** 2)

    def test_r2_never_exceeds_r2_opt(self):
        for k in (0.2, 0.5, 0.8, 1.0, 1.4):
            m = placement_metrics(_frame(rho=0.7, k=k, b=0.2), "y")
            assert m["r2"] <= m["r2_opt"] + 1e-12
            assert m["gap"] >= 0.0


class TestKnownShapes:
    def test_calibrated_predictor_has_no_gap(self):
        """Squared loss is optimal at k == rho, not k == 1, so that predictor sits at ceiling."""
        m = placement_metrics(_frame(rho=0.8, k=0.8, b=0.0), "y")
        assert m["gap"] == pytest.approx(0.0, abs=1e-9)
        assert m["r2"] == pytest.approx(m["r2_opt"])

    def test_full_spread_is_worse_than_shrunk(self):
        """k == 1 looks 'honest' but costs (rho - 1)**2 against the shrunk optimum."""
        shrunk = placement_metrics(_frame(rho=0.8, k=0.8, b=0.0), "y")
        full = placement_metrics(_frame(rho=0.8, k=1.0, b=0.0), "y")
        assert full["gap"] > shrunk["gap"]
        assert full["gap"] == pytest.approx((0.8 - 1.0) ** 2)

    def test_pure_offset_puts_the_gap_in_bias(self):
        m = placement_metrics(_frame(rho=0.8, k=0.8, b=0.5), "y")
        assert m["bias"] == pytest.approx(0.5)
        assert m["spread_ratio"] == pytest.approx(0.8)
        assert m["gap"] == pytest.approx(0.5**2)

    def test_pure_shrinkage_puts_the_gap_in_spread(self):
        m = placement_metrics(_frame(rho=0.8, k=0.4, b=0.0), "y")
        assert m["bias"] == pytest.approx(0.0, abs=1e-9)
        assert m["spread_ratio"] == pytest.approx(0.4)
        assert m["gap"] == pytest.approx((0.8 - 0.4) ** 2)

    def test_offset_leaves_ranking_untouched(self):
        """A constant shift cannot reorder, so the ceiling is identical and only r2 moves."""
        clean = placement_metrics(_frame(rho=0.8, k=0.8, b=0.0), "y")
        shifted = placement_metrics(_frame(rho=0.8, k=0.8, b=0.5), "y")
        assert shifted["r2_opt"] == pytest.approx(clean["r2_opt"])
        assert shifted["r2"] < clean["r2"]


class TestFrameHandling:
    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="nope"):
            placement_metrics(_frame(0.8, 0.8, 0.0), "nope")

    def test_missing_prediction_column_raises(self):
        with pytest.raises(ValueError, match="y_pred"):
            placement_metrics(_frame(0.8, 0.8, 0.0), "y", prediction_col="y_pred")

    def test_multi_target_prediction_column(self):
        """A multi-target frame names its columns <target>_pred; the bare one is target[0]."""
        df = _frame(rho=0.8, k=0.8, b=0.5).rename(columns={"prediction": "y_pred"})
        m = placement_metrics(df, "y", prediction_col="y_pred")
        assert m["bias"] == pytest.approx(0.5)

    def test_nan_pairs_are_dropped(self):
        df = _frame(rho=0.8, k=0.8, b=0.5)
        clean = placement_metrics(df, "y")
        df.loc[:4, "y"] = np.nan
        df.loc[5:9, "prediction"] = np.nan
        dropped = placement_metrics(df, "y")
        # Same underlying relationship, so the terms survive the 10 removed rows
        assert dropped["bias"] == pytest.approx(clean["bias"], abs=0.05)
        assert dropped["spread_ratio"] == pytest.approx(clean["spread_ratio"], abs=0.05)

    def test_all_nan_returns_empty(self):
        df = pd.DataFrame({"y": [np.nan] * 4, "prediction": [1.0, 2.0, 3.0, 4.0]})
        assert placement_metrics(df, "y") == {}

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_single_row_has_no_correlation(self):
        df = pd.DataFrame({"y": [5.0], "prediction": [4.0]})
        m = placement_metrics(df, "y")
        assert m["bias"] == pytest.approx(-1.0)
        assert np.isnan(m["r2_opt"])
        assert np.isnan(m["gap"])
