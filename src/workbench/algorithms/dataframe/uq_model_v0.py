"""UQModelV0: isotonic-on-(prediction, std) regression confidence calibrator.

The lightweight counterpart to :class:`UQModelV1`. No proximity, no neighborhood
features — inputs are just ``(prediction, prediction_std)``. Useful as the
default for models without a SMILES column or when you want a fast, easy-to-
audit calibrator that doesn't depend on a similarity index.

Algorithm:

    Calibration  (y_val, y_pred, prediction_std):
        1. Bin y_pred into N=10 quantile bins.
        2. Within each bin, fit IsotonicRegression(std -> |residual|),
           falling back to a global isotonic for bins with <20 samples.
        3. Apply the calibrator back on the cal set to get expected_residual_cal.
        4. Store 0..100 percentiles of that distribution.
        5. Also fit split-conformal scale factors q_alpha for each target
           coverage level: q = quantile_{(1+1/n)*alpha} of |residual| / std.

    Inference  (prediction, prediction_std):
        1. Look up the prediction's bin, apply that bin's isotonic to get
           expected_residual.
        2. confidence = 1 - percentile_rank(expected_residual)  (in [0, 1]).
        3. Intervals: prediction +/- q_alpha * std.

References:
    - Lei et al. 2018, "Distribution-Free Predictive Inference for Regression"
      (locally adaptive conformal prediction via a learned scale function).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import logging

log = logging.getLogger("workbench")


# Defaults preserved from v0.8.338
DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]
DEFAULT_RESIDUAL_CALIBRATOR_BINS = 10
MIN_SAMPLES_PER_BIN = 20

# Quantile column names by confidence level — matched to the current UQModel
# output so v0 and current can be compared row-for-row.
_QUANTILE_COLUMNS = {
    0.50: ("q_25", "q_75"),
    0.68: ("q_16", "q_84"),
    0.80: ("q_10", "q_90"),
    0.90: ("q_05", "q_95"),
    0.95: ("q_025", "q_975"),
}


# =============================================================================
# Internal calibrator (per-bin isotonic) — preserved verbatim from v0.8.338
# =============================================================================
def _fit_residual_calibrator(
    y_pred: np.ndarray,
    prediction_std: np.ndarray,
    abs_residual: np.ndarray,
    n_bins: int = DEFAULT_RESIDUAL_CALIBRATOR_BINS,
) -> dict:
    """Fit a residual-aware calibrator: (prediction, std) -> expected |residual|.

    Bin predictions into quantile bins, fit IsotonicRegression(std -> |residual|)
    within each bin, fall back to a global fit for under-populated bins. Stored
    as piecewise-linear thresholds for sklearn-free inference (np.interp).
    """
    from sklearn.isotonic import IsotonicRegression

    y_pred = np.asarray(y_pred).flatten()
    prediction_std = np.asarray(prediction_std).flatten()
    abs_residual = np.asarray(abs_residual).flatten()

    quantile_points = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_pred, quantile_points))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    global_iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
    global_iso.fit(prediction_std, abs_residual)

    isotonic_bins = []
    for i in range(len(bin_edges) - 1):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() < MIN_SAMPLES_PER_BIN:
            iso = global_iso
        else:
            iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
            iso.fit(prediction_std[mask], abs_residual[mask])
        isotonic_bins.append(
            {
                "x_thresholds": iso.X_thresholds_.tolist(),
                "y_thresholds": iso.y_thresholds_.tolist(),
            }
        )

    return {
        "prediction_bin_edges": bin_edges.tolist(),
        "isotonic_bins": isotonic_bins,
    }


def _apply_residual_calibrator(
    predictions: np.ndarray,
    stds: np.ndarray,
    calibrator: dict,
) -> np.ndarray:
    """Apply a stored residual calibrator to (prediction, std) pairs."""
    bin_edges = np.asarray(calibrator["prediction_bin_edges"])
    isotonic_bins = calibrator["isotonic_bins"]
    n_bins = len(isotonic_bins)

    predictions = np.asarray(predictions).flatten()
    stds = np.asarray(stds).flatten()

    bin_idx = np.clip(np.searchsorted(bin_edges, predictions, side="right") - 1, 0, n_bins - 1)

    expected_residual = np.empty(len(predictions), dtype=float)
    for i, iso in enumerate(isotonic_bins):
        mask = bin_idx == i
        if not mask.any():
            continue
        x_thr = np.asarray(iso["x_thresholds"])
        y_thr = np.asarray(iso["y_thresholds"])
        expected_residual[mask] = np.interp(stds[mask], x_thr, y_thr)

    return np.clip(expected_residual, 0.0, None)


# =============================================================================
# UQModelV0
# =============================================================================
class UQModelV0:
    """Isotonic-on-(prediction, std) regression UQ pipeline.

    Companion to :class:`workbench.algorithms.dataframe.uq_model_v1.UQModelV1`;
    the two share a ``.predict(query, predictions, prediction_std)`` signature
    so they can be used interchangeably.

    Differences from UQModelV1:
        * No proximity backend, no neighborhood features.
        * Calibrator is a per-bin isotonic over (prediction, std), not a
          RandomForest over (prediction, std, knn_distance, knn_target_std,
          local_pred_gap).
        * ``query`` argument is accepted for signature compatibility but used
          only to label the result DataFrame's index — V0 has no id-lookup
          because it has no reference index.

    Usage:
        uq0 = UQModelV0.fit(y_val, y_pred_val, prediction_std_val)
        out = uq0.predict(ids, predictions, prediction_std)

        # Save / load (uq_metadata_v0.json)
        uq0.save(model_dir)
        uq0 = UQModelV0.load(model_dir)
    """

    METADATA_FILENAME = "uq_metadata_v0.json"
    UQ_VERSION = "v0"

    def __init__(
        self,
        confidence_levels: List[float],
        scale_factors: dict,
        residual_calibrator: dict,
        residual_percentiles: List[float],
    ):
        self.confidence_levels = list(confidence_levels)
        self.scale_factors = dict(scale_factors)
        self.residual_calibrator = residual_calibrator
        self.residual_percentiles = list(residual_percentiles)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        y_val: Union[np.ndarray, pd.Series],
        y_pred_val: Union[np.ndarray, pd.Series],
        prediction_std_val: Union[np.ndarray, pd.Series],
        confidence_levels: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> "UQModelV0":
        """Fit the v0 calibrator on validation predictions.

        Args:
            y_val: True target values, shape (n,).
            y_pred_val: Predicted values from the model, shape (n,).
            prediction_std_val: Ensemble std for each prediction, shape (n,).
            confidence_levels: Target coverage levels for prediction intervals.
                Default: [0.50, 0.68, 0.80, 0.90, 0.95].
            verbose: If True, print per-level scale factor + empirical coverage.

        Returns:
            A fitted UQModelV0.
        """
        if confidence_levels is None:
            confidence_levels = DEFAULT_CONFIDENCE_LEVELS

        y_val = np.asarray(y_val, dtype=float).flatten()
        y_pred_val = np.asarray(y_pred_val, dtype=float).flatten()
        prediction_std_val = np.asarray(prediction_std_val, dtype=float).flatten()

        safe_std = np.maximum(prediction_std_val, 1e-10)
        nonconformity_scores = np.abs(y_val - y_pred_val) / safe_std

        scale_factors = {}
        if verbose:
            log.info("Calibrating prediction intervals (v0) from ensemble std...")
            log.info(f"  Validation samples: {len(y_val)}")
            log.info(f"  Mean ensemble std: {np.mean(prediction_std_val):.4f}")
            log.info(f"  Median ensemble std: {np.median(prediction_std_val):.4f}")

        for confidence_level in confidence_levels:
            n = len(nonconformity_scores)
            adjusted_quantile = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
            q = float(np.quantile(nonconformity_scores, adjusted_quantile))
            scale_factors[f"{confidence_level:.2f}"] = q
            if verbose:
                lower = y_pred_val - q * safe_std
                upper = y_pred_val + q * safe_std
                coverage = np.mean((y_val >= lower) & (y_val <= upper))
                log.info(f"  {confidence_level * 100:.0f}% CI: scale_factor={q:.3f}, coverage={coverage * 100:.1f}%")

        # Per-bin isotonic calibrator + reference percentile distribution
        abs_residual = np.abs(y_val - y_pred_val)
        residual_calibrator = _fit_residual_calibrator(y_pred_val, prediction_std_val, abs_residual)
        expected_residual_cal = _apply_residual_calibrator(y_pred_val, prediction_std_val, residual_calibrator)
        residual_percentiles = [float(np.percentile(expected_residual_cal, p)) for p in range(101)]

        if verbose:
            log.info("Residual-aware confidence calibrator fit (v0):")
            log.info(f"  Prediction bins: {len(residual_calibrator['isotonic_bins'])}")
            log.info(
                f"  Expected |residual| on cal set: "
                f"min={expected_residual_cal.min():.4f}, "
                f"median={np.median(expected_residual_cal):.4f}, "
                f"max={expected_residual_cal.max():.4f}"
            )

        return cls(
            confidence_levels=confidence_levels,
            scale_factors=scale_factors,
            residual_calibrator=residual_calibrator,
            residual_percentiles=residual_percentiles,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        query: Optional[Union[List, pd.Series, np.ndarray, pd.DataFrame]],
        predictions: Union[np.ndarray, pd.Series],
        prediction_std: Union[np.ndarray, pd.Series],
    ) -> pd.DataFrame:
        """Compute v0 UQ outputs (expected residual, confidence, intervals).

        The ``query`` argument is accepted for signature compatibility with
        :class:`UQModelV1` but used only to derive the result DataFrame's index
        — v0 has no proximity backend, so the query payload itself isn't needed
        for the calibration math.

        Args:
            query: IDs (list/Series/array) or a DataFrame whose index will be
                propagated onto the result. ``None`` falls back to a default
                RangeIndex.
            predictions: Model predictions (ensemble mean), shape (n,).
            prediction_std: Ensemble standard deviation, shape (n,).

        Returns:
            DataFrame with columns:
                expected_residual, confidence, q_025, q_05, q_10, q_16, q_25,
                q_50, q_75, q_84, q_90, q_95, q_975
        """
        predictions = np.asarray(predictions, dtype=float).flatten()
        prediction_std = np.asarray(prediction_std, dtype=float).flatten()
        if len(predictions) != len(prediction_std):
            raise ValueError(
                f"predictions length ({len(predictions)}) must match prediction_std length ({len(prediction_std)})"
            )

        # Derive the result index from the query argument (V1 compatibility).
        if isinstance(query, pd.DataFrame):
            index = query.index
        elif query is None:
            index = np.arange(len(predictions))
        else:
            index = list(query)
            if len(index) != len(predictions):
                raise ValueError(
                    f"query length ({len(index)}) must match predictions length ({len(predictions)})"
                )

        safe_std = np.maximum(prediction_std, 1e-10)

        expected_residual = _apply_residual_calibrator(predictions, prediction_std, self.residual_calibrator)

        residual_percentiles = np.asarray(self.residual_percentiles)
        ranks = np.searchsorted(residual_percentiles, expected_residual, side="right") / len(residual_percentiles)
        confidence = np.clip(1.0 - ranks, 0.0, 1.0)

        result = pd.DataFrame(
            {
                "expected_residual": expected_residual,
                "confidence": confidence,
                "q_50": predictions,
            },
            index=index,
        )

        for alpha in self.confidence_levels:
            q = self.scale_factors[f"{alpha:.2f}"]
            lower = predictions - q * safe_std
            upper = predictions + q * safe_std
            if alpha in _QUANTILE_COLUMNS:
                lo_col, hi_col = _QUANTILE_COLUMNS[alpha]
                result[lo_col] = lower
                result[hi_col] = upper

        quantile_cols = ["q_025", "q_05", "q_10", "q_16", "q_25", "q_50", "q_75", "q_84", "q_90", "q_95", "q_975"]
        existing_q = [c for c in quantile_cols if c in result.columns]
        return result[["expected_residual", "confidence"] + existing_q]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "confidence_levels": list(self.confidence_levels),
            "scale_factors": dict(self.scale_factors),
            "residual_calibrator": self.residual_calibrator,
            "residual_percentiles": list(self.residual_percentiles),
        }

    def save(self, model_dir: str, filename: Optional[str] = None) -> None:
        """Save to JSON.

        Args:
            model_dir: Directory to save the metadata file.
            filename: Override the filename (default: ``uq_metadata_v0.json``).
                Pass ``"uq_metadata.json"`` to write a file that the old
                v0.8.338 model_script_utils.uq_harness loaders accept verbatim.
        """
        path = os.path.join(model_dir, filename or self.METADATA_FILENAME)
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=2)
        log.info(f"Saved UQModelV0 metadata to {path}")

    @classmethod
    def from_dict(cls, metadata: dict) -> "UQModelV0":
        """Reconstruct from a metadata dict."""
        return cls(
            confidence_levels=metadata["confidence_levels"],
            scale_factors=metadata["scale_factors"],
            residual_calibrator=metadata["residual_calibrator"],
            residual_percentiles=metadata["residual_percentiles"],
        )

    @classmethod
    def load(cls, model_dir: str, filename: Optional[str] = None) -> "UQModelV0":
        """Load from JSON (defaults to ``uq_metadata_v0.json``)."""
        path = os.path.join(model_dir, filename or cls.METADATA_FILENAME)
        with open(path) as fp:
            return cls.from_dict(json.load(fp))
