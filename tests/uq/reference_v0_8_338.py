"""Vendored UQ harness from workbench v0.8.338 (git tag ``v0.8.338``, commit
``dc8be9477``).

This is a frozen golden reference. The functions below are byte-faithful copies
of the regression-UQ surface from ``src/workbench/model_script_utils/uq_harness.py``
at that tag, with three intentional adjustments:

  1. The classification-UQ surface (``compute_vgmu_confidence``,
     ``calibrate_classification_confidence``, ``apply_classification_confidence``)
     is omitted — outside V0's scope.
  2. ``print()`` calls are kept verbatim. They are behaviorally inert; the test
     runner can capture stdout if it cares.
  3. The module docstring has been replaced with this header so the file is
     clearly labeled as vendored history.

DO NOT MODIFY THIS FILE. Its purpose is to be a fixed equivalence baseline for
``UQModelV0``. If we ever need to update what "the v0.8.338 algorithm" means,
that decision is upstream of this file and warrants its own discussion.
"""

import json
import os
import numpy as np
import pandas as pd

# Default confidence levels for prediction intervals
DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]

# Default number of prediction bins for the residual-aware calibrator
DEFAULT_RESIDUAL_CALIBRATOR_BINS = 10
MIN_SAMPLES_PER_BIN = 20


def _fit_residual_calibrator(
    y_pred: np.ndarray,
    prediction_std: np.ndarray,
    abs_residual: np.ndarray,
    n_bins: int = DEFAULT_RESIDUAL_CALIBRATOR_BINS,
) -> dict:
    """Fit a residual-aware calibrator: (prediction, std) -> expected |residual|."""
    from sklearn.isotonic import IsotonicRegression

    y_pred = np.asarray(y_pred).flatten()
    prediction_std = np.asarray(prediction_std).flatten()
    abs_residual = np.asarray(abs_residual).flatten()

    # Quantile-based bin edges on prediction; collapse duplicates from ties
    quantile_points = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_pred, quantile_points))
    # Expand outer bounds so all cal-set predictions strictly fall inside
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    # Per-bin isotonic fits with a small-sample fallback to a global fit
    isotonic_bins = []
    global_iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
    global_iso.fit(prediction_std, abs_residual)

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

    # Locate each prediction in its bin (clipped to valid range for out-of-cal preds)
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


def calibrate_uq(
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    prediction_std_val: np.ndarray,
    confidence_levels=None,
) -> dict:
    """Calibrate prediction intervals using conformal prediction on ensemble std."""
    if confidence_levels is None:
        confidence_levels = DEFAULT_CONFIDENCE_LEVELS

    y_val = np.asarray(y_val).flatten()
    y_pred_val = np.asarray(y_pred_val).flatten()
    prediction_std_val = np.asarray(prediction_std_val).flatten()

    # Compute nonconformity scores: |residual| / std
    # For samples with zero std (all ensemble members agree perfectly), use a small epsilon
    safe_std = np.maximum(prediction_std_val, 1e-10)
    nonconformity_scores = np.abs(y_val - y_pred_val) / safe_std

    # For each confidence level, find the scaling factor (quantile of nonconformity scores)
    # that achieves the target coverage
    scale_factors = {}
    print("\nCalibrating prediction intervals from ensemble std...")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Mean ensemble std: {np.mean(prediction_std_val):.4f}")
    print(f"  Median ensemble std: {np.median(prediction_std_val):.4f}")

    for confidence_level in confidence_levels:
        # Conformal quantile: use (1 - alpha) quantile of nonconformity scores
        # with finite-sample correction: ceil((n+1) * confidence_level) / n
        n = len(nonconformity_scores)
        adjusted_quantile = min(np.ceil((n + 1) * confidence_level) / n, 1.0)
        q = float(np.quantile(nonconformity_scores, adjusted_quantile))
        scale_factors[f"{confidence_level:.2f}"] = q

        # Validate coverage
        lower = y_pred_val - q * safe_std
        upper = y_pred_val + q * safe_std
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        print(f"  {confidence_level * 100:.0f}% CI: scale_factor={q:.3f}, coverage={coverage * 100:.1f}%")

    # Compute interval width analysis
    print("\nInterval Width Analysis:")
    for confidence_level in confidence_levels:
        q = scale_factors[f"{confidence_level:.2f}"]
        widths = 2 * q * safe_std
        print(f"  {confidence_level * 100:.0f}% CI: Mean width={np.mean(widths):.3f}, Std={np.std(widths):.3f}")

    # Fit the residual-aware calibrator and its reference percentile distribution.
    abs_residual = np.abs(y_val - y_pred_val)
    residual_calibrator = _fit_residual_calibrator(y_pred_val, prediction_std_val, abs_residual)
    expected_residual_cal = _apply_residual_calibrator(y_pred_val, prediction_std_val, residual_calibrator)
    residual_percentiles = [float(np.percentile(expected_residual_cal, p)) for p in range(101)]

    print("\nResidual-aware confidence calibrator fit:")
    print(f"  Prediction bins: {len(residual_calibrator['isotonic_bins'])}")
    print(
        f"  Expected |residual| on cal set: "
        f"min={expected_residual_cal.min():.4f}, "
        f"median={np.median(expected_residual_cal):.4f}, "
        f"max={expected_residual_cal.max():.4f}"
    )

    uq_metadata = {
        "confidence_levels": confidence_levels,
        "scale_factors": scale_factors,
        "residual_calibrator": residual_calibrator,
        "residual_percentiles": residual_percentiles,
    }

    return uq_metadata


def save_uq_metadata(uq_metadata: dict, model_dir: str) -> None:
    """Save UQ metadata to disk."""
    with open(os.path.join(model_dir, "uq_metadata.json"), "w") as fp:
        json.dump(uq_metadata, fp, indent=2)
    print(f"Saved UQ metadata to {model_dir}")


def load_uq_metadata(model_dir: str) -> dict:
    """Load UQ metadata from disk."""
    uq_metadata_path = os.path.join(model_dir, "uq_metadata.json")
    with open(uq_metadata_path) as fp:
        uq_metadata = json.load(fp)
    return uq_metadata


def predict_intervals(
    df: pd.DataFrame,
    uq_metadata: dict,
    prediction_col: str = "prediction",
    std_col: str = "prediction_std",
) -> pd.DataFrame:
    """Add calibrated prediction intervals to a DataFrame."""
    predictions = df[prediction_col].values
    prediction_std = df[std_col].values
    safe_std = np.maximum(prediction_std, 1e-10)

    confidence_levels = uq_metadata["confidence_levels"]
    scale_factors = uq_metadata["scale_factors"]

    # Confidence level -> quantile column name mapping
    quantile_map = {
        0.50: ("q_25", "q_75"),
        0.68: ("q_16", "q_84"),
        0.80: ("q_10", "q_90"),
        0.90: ("q_05", "q_95"),
        0.95: ("q_025", "q_975"),
    }

    for conf_level in confidence_levels:
        q = scale_factors[f"{conf_level:.2f}"]
        lower = predictions - q * safe_std
        upper = predictions + q * safe_std

        if conf_level in quantile_map:
            lower_col, upper_col = quantile_map[conf_level]
            df[lower_col] = lower
            df[upper_col] = upper

    # Set q_50 as the prediction itself (median of the ensemble)
    df["q_50"] = predictions

    # Reorder quantile columns for easier reading
    quantile_cols = ["q_025", "q_05", "q_10", "q_16", "q_25", "q_50", "q_75", "q_84", "q_90", "q_95", "q_975"]
    existing_q_cols = [c for c in quantile_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in quantile_cols]
    df = df[other_cols + existing_q_cols]

    return df


def compute_confidence(
    df: pd.DataFrame,
    uq_metadata: dict,
    std_col: str = "prediction_std",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute confidence scores (0.0 to 1.0) for each prediction."""
    predictions = df[prediction_col].values
    std_values = df[std_col].abs().values

    expected_residual = _apply_residual_calibrator(predictions, std_values, uq_metadata["residual_calibrator"])
    residual_percentiles = np.asarray(uq_metadata["residual_percentiles"])
    percentile_ranks = np.searchsorted(residual_percentiles, expected_residual, side="right") / len(
        residual_percentiles
    )

    df.loc[:, "confidence"] = np.clip(1.0 - percentile_ranks, 0.0, 1.0)

    return df
