"""UQ Harness: Uncertainty Quantification via Conformalized Ensemble Uncertainty.

This module provides calibrated prediction intervals by combining ensemble disagreement
(prediction_std) with conformal calibration. The approach works uniformly across all
model types (XGBoost, PyTorch, ChemProp).

How it works:
    1. Each model framework trains a K-fold ensemble and computes prediction_std
       (standard deviation across ensemble members) — this captures the model's own
       uncertainty signal.
    2. This harness calibrates that raw std into prediction intervals with guaranteed
       coverage using conformal prediction:
       - For each confidence level, find a scaling factor `q` such that
         `prediction ± q * prediction_std` covers the target percentage on held-out data.
       - This is a split-conformal approach: the scaling factors are computed on validation
         data that was NOT used for training.

Why ensemble std + conformal calibration?
    - Ensemble disagreement is the best available uncertainty signal — it comes from
      the model itself. When ensemble members disagree on a prediction, the actual error
      tends to be larger (high interval-to-error correlation).
    - Raw ensemble std is poorly calibrated (intervals are often too narrow or too wide).
      Conformal calibration fixes this with distribution-free coverage guarantees.

Usage:
    # Training: calibrate prediction intervals from ensemble std
    uq_metadata = calibrate_uq(y_val, y_pred_val, prediction_std_val)
    save_uq_metadata(uq_metadata, model_dir)

    # Inference: apply calibrated intervals to new predictions
    uq_metadata = load_uq_metadata(model_dir)
    df = predict_intervals(df)
    df = compute_confidence(df, uq_metadata)
"""

import json
import os
import numpy as np
import pandas as pd

# Default confidence levels for prediction intervals
DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]


def calibrate_uq(
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    prediction_std_val: np.ndarray,
    confidence_levels: list[float] | None = None,
) -> dict:
    """Calibrate prediction intervals using conformal prediction on ensemble std.

    Computes scaling factors so that `prediction ± scale_factor * prediction_std`
    achieves the target coverage on validation data. This is a split-conformal
    approach with distribution-free coverage guarantees.

    Args:
        y_val (np.ndarray): True target values (validation set)
        y_pred_val (np.ndarray): Predicted values (validation set)
        prediction_std_val (np.ndarray): Ensemble std values (validation set)
        confidence_levels (list[float]): Confidence levels (default: [0.50, 0.68, 0.80, 0.90, 0.95])

    Returns:
        dict: UQ metadata with scale_factors, std_percentiles, and confidence_levels
    """
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

    # Store the std distribution for percentile-rank confidence scoring
    # 101 percentile values (0th, 1st, ..., 100th) for smooth interpolation at inference
    std_percentiles = [float(np.percentile(prediction_std_val, p)) for p in range(101)]

    # Compute interval width analysis
    print("\nInterval Width Analysis:")
    for confidence_level in confidence_levels:
        q = scale_factors[f"{confidence_level:.2f}"]
        widths = 2 * q * safe_std
        print(f"  {confidence_level * 100:.0f}% CI: Mean width={np.mean(widths):.3f}, Std={np.std(widths):.3f}")

    uq_metadata = {
        "confidence_levels": confidence_levels,
        "scale_factors": scale_factors,
        "std_percentiles": std_percentiles,
    }

    return uq_metadata


def save_uq_metadata(uq_metadata: dict, model_dir: str) -> None:
    """Save UQ metadata to disk.

    Args:
        uq_metadata (dict): UQ metadata from calibrate_uq()
        model_dir (str): Directory to save metadata
    """
    with open(os.path.join(model_dir, "uq_metadata.json"), "w") as fp:
        json.dump(uq_metadata, fp, indent=2)

    print(f"Saved UQ metadata to {model_dir}")


def load_uq_metadata(model_dir: str) -> dict:
    """Load UQ metadata from disk.

    Args:
        model_dir (str): Directory containing saved metadata

    Returns:
        dict: UQ metadata with scale_factors, std_percentiles, and confidence_levels
    """
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
    """Add calibrated prediction intervals to a DataFrame.

    Uses the conformally-calibrated scaling factors to convert ensemble std
    into prediction intervals with guaranteed coverage.

    Interval: prediction ± scale_factor * prediction_std

    Args:
        df (pd.DataFrame): DataFrame with prediction and prediction_std columns
        uq_metadata (dict): UQ metadata from calibrate_uq() or load_uq_metadata()
        prediction_col (str): Name of the prediction column (default: 'prediction')
        std_col (str): Name of the prediction_std column (default: 'prediction_std')

    Returns:
        pd.DataFrame: DataFrame with added quantile columns (q_025, q_05, ..., q_975)
    """
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

    # Calculate pseudo-standard deviation from the 68% interval width
    if "q_84" in df.columns and "q_16" in df.columns:
        df["prediction_std"] = (df["q_84"] - df["q_16"]).abs() / 2.0

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
) -> pd.DataFrame:
    """Compute confidence scores (0.0 to 1.0) based on ensemble prediction std.

    Uses percentile-rank of prediction_std against the calibration distribution:
    - confidence = 1 - percentile_rank(std)
    - Low std (ensemble agreement) → high percentile rank → high confidence
    - High std (ensemble disagreement) → low percentile rank → low confidence

    Interpretation: confidence of 0.7 means this prediction's uncertainty is lower
    than 70% of predictions in the calibration set.

    Args:
        df (pd.DataFrame): DataFrame with prediction_std column
        uq_metadata (dict): UQ metadata containing std_percentiles
        std_col (str): Name of the std column (default: 'prediction_std')

    Returns:
        pd.DataFrame: DataFrame with added 'confidence' column (values between 0 and 1)
    """
    std_percentiles = np.array(uq_metadata["std_percentiles"])
    std_values = df[std_col].abs().values

    # For each prediction's std, find where it falls in the calibration distribution
    # np.searchsorted gives the index where each value would be inserted to maintain order
    # Dividing by 100 (len - 1) gives us the percentile rank (0.0 to 1.0)
    percentile_ranks = np.searchsorted(std_percentiles, std_values, side="right") / len(std_percentiles)

    # Confidence = 1 - percentile_rank (low std = high confidence)
    # Clip to [0, 1] for predictions outside the calibration range
    df["confidence"] = np.clip(1.0 - percentile_ranks, 0.0, 1.0)

    return df
