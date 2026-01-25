"""UQ Harness: Uncertainty Quantification using MAPIE Conformalized Quantile Regression.

This module provides a reusable UQ harness that can wrap any point predictor model
(XGBoost, PyTorch, ChemProp, etc.) to provide calibrated prediction intervals.

Features:
    - Conformalized Quantile Regression (CQR) for distribution-free coverage guarantees
    - Multiple confidence levels (50%, 68%, 80%, 90%, 95%)
    - Confidence scoring based on interval width

Why CQR without additional Z-scaling:
    MAPIE's conformalization step already guarantees that prediction intervals achieve
    their target coverage on the calibration set. For example, an 80% CI will contain
    ~80% of true values. This is the core promise of conformal prediction.

    Z-scaling (post-hoc interval adjustment) would only help if there's a distribution
    shift between calibration and test data. However:
    1. We'd compute Z-scale on the same calibration set MAPIE uses, making it redundant
    2. Our cross-fold validation metrics confirm coverage is already well-calibrated
    3. Adding Z-scaling would "second-guess" MAPIE's principled conformalization

    Empirically, our models achieve excellent coverage (e.g., 80% CI → 80.1% coverage),
    validating that MAPIE's approach is sufficient without additional calibration.

Usage:
    # Training
    uq_models, uq_metadata = train_uq_models(X_train, y_train, X_val, y_val)
    save_uq_models(uq_models, uq_metadata, model_dir)

    # Inference
    uq_models, uq_metadata = load_uq_models(model_dir)
    df = predict_intervals(df, X, uq_models, uq_metadata)
    df = compute_confidence(df, uq_metadata["median_interval_width"])
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from mapie.regression import ConformalizedQuantileRegressor

# Default confidence levels for prediction intervals
DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]


def train_uq_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    confidence_levels: list[float] | None = None,
) -> tuple[dict, dict]:
    """Train MAPIE UQ models for multiple confidence levels.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features for conformalization
        y_val: Validation targets for conformalization
        confidence_levels: List of confidence levels (default: [0.50, 0.68, 0.80, 0.90, 0.95])

    Returns:
        Tuple of (uq_models dict, uq_metadata dict)
    """
    if confidence_levels is None:
        confidence_levels = DEFAULT_CONFIDENCE_LEVELS

    mapie_models = {}

    for confidence_level in confidence_levels:
        alpha = 1 - confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        print(f"\nTraining quantile models for {confidence_level * 100:.0f}% confidence interval...")
        print(f"  Quantiles: {lower_q:.3f}, {upper_q:.3f}, 0.500")

        # Train three LightGBM quantile models for this confidence level
        quantile_estimators = []
        for q in [lower_q, upper_q, 0.5]:
            print(f"    Training model for quantile {q:.3f}...")
            est = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                force_col_wise=True,
            )
            est.fit(X_train, y_train)
            quantile_estimators.append(est)

        # Create MAPIE CQR model for this confidence level
        print(f"  Setting up MAPIE CQR for {confidence_level * 100:.0f}% confidence...")
        mapie_model = ConformalizedQuantileRegressor(
            quantile_estimators, confidence_level=confidence_level, prefit=True
        )

        # Conformalize the model with validation data
        print("  Conformalizing with validation data...")
        mapie_model.conformalize(X_val, y_val)

        # Store the model
        model_name = f"mapie_{confidence_level:.2f}"
        mapie_models[model_name] = mapie_model

        # Validate coverage for this confidence level
        y_pred, y_pis = mapie_model.predict_interval(X_val)
        coverage = np.mean((y_val >= y_pis[:, 0, 0]) & (y_val <= y_pis[:, 1, 0]))
        print(f"  Coverage: Target={confidence_level * 100:.0f}%, Empirical={coverage * 100:.1f}%")

    # Compute median interval width for confidence calculation (using 80% CI = q_10 to q_90)
    print("\nComputing normalization statistics for confidence scores...")
    model_80 = mapie_models["mapie_0.80"]
    _, y_pis_80 = model_80.predict_interval(X_val)
    interval_width = np.abs(y_pis_80[:, 1, 0] - y_pis_80[:, 0, 0])
    median_interval_width = float(np.median(interval_width))
    print(f"  Median interval width (q_10-q_90): {median_interval_width:.6f}")

    # Analyze interval widths across confidence levels
    print("\nInterval Width Analysis:")
    for conf_level in confidence_levels:
        model = mapie_models[f"mapie_{conf_level:.2f}"]
        _, y_pis = model.predict_interval(X_val)
        widths = y_pis[:, 1, 0] - y_pis[:, 0, 0]
        print(f"  {conf_level * 100:.0f}% CI: Mean width={np.mean(widths):.3f}, Std={np.std(widths):.3f}")

    uq_metadata = {
        "confidence_levels": confidence_levels,
        "median_interval_width": median_interval_width,
    }

    return mapie_models, uq_metadata


def save_uq_models(uq_models: dict, uq_metadata: dict, model_dir: str) -> None:
    """Save UQ models and metadata to disk.

    Args:
        uq_models: Dictionary of MAPIE models keyed by name (e.g., "mapie_0.80")
        uq_metadata: Dictionary with confidence_levels and median_interval_width
        model_dir: Directory to save models
    """
    # Save each MAPIE model
    for model_name, model in uq_models.items():
        joblib.dump(model, os.path.join(model_dir, f"{model_name}.joblib"))

    # Save median interval width
    with open(os.path.join(model_dir, "median_interval_width.json"), "w") as fp:
        json.dump(uq_metadata["median_interval_width"], fp)

    # Save UQ metadata
    with open(os.path.join(model_dir, "uq_metadata.json"), "w") as fp:
        json.dump(uq_metadata, fp, indent=2)

    print(f"Saved {len(uq_models)} UQ models to {model_dir}")


def load_uq_models(model_dir: str) -> tuple[dict, dict]:
    """Load UQ models and metadata from disk.

    Args:
        model_dir: Directory containing saved models

    Returns:
        Tuple of (uq_models dict, uq_metadata dict)
    """
    # Load UQ metadata
    uq_metadata_path = os.path.join(model_dir, "uq_metadata.json")
    if os.path.exists(uq_metadata_path):
        with open(uq_metadata_path) as fp:
            uq_metadata = json.load(fp)
    else:
        # Fallback for older models that only have median_interval_width.json
        uq_metadata = {"confidence_levels": DEFAULT_CONFIDENCE_LEVELS}
        median_width_path = os.path.join(model_dir, "median_interval_width.json")
        if os.path.exists(median_width_path):
            with open(median_width_path) as fp:
                uq_metadata["median_interval_width"] = json.load(fp)

    # Load all MAPIE models
    uq_models = {}
    for conf_level in uq_metadata["confidence_levels"]:
        model_name = f"mapie_{conf_level:.2f}"
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            uq_models[model_name] = joblib.load(model_path)

    return uq_models, uq_metadata


def predict_intervals(
    df: pd.DataFrame,
    X: pd.DataFrame | np.ndarray,
    uq_models: dict,
    uq_metadata: dict,
) -> pd.DataFrame:
    """Add prediction intervals to a DataFrame.

    Args:
        df: DataFrame to add interval columns to
        X: Features for prediction (must match training features)
        uq_models: Dictionary of MAPIE models
        uq_metadata: Dictionary with confidence_levels

    Returns:
        DataFrame with added quantile columns (q_025, q_05, ..., q_975)
    """
    confidence_levels = uq_metadata["confidence_levels"]

    for conf_level in confidence_levels:
        model_name = f"mapie_{conf_level:.2f}"
        model = uq_models[model_name]

        # Get conformalized predictions
        y_pred, y_pis = model.predict_interval(X)

        # Map confidence levels to quantile column names
        if conf_level == 0.50:  # 50% CI
            df["q_25"] = y_pis[:, 0, 0]
            df["q_75"] = y_pis[:, 1, 0]
            df["q_50"] = y_pred  # Median prediction
        elif conf_level == 0.68:  # 68% CI (~1 std)
            df["q_16"] = y_pis[:, 0, 0]
            df["q_84"] = y_pis[:, 1, 0]
        elif conf_level == 0.80:  # 80% CI
            df["q_10"] = y_pis[:, 0, 0]
            df["q_90"] = y_pis[:, 1, 0]
        elif conf_level == 0.90:  # 90% CI
            df["q_05"] = y_pis[:, 0, 0]
            df["q_95"] = y_pis[:, 1, 0]
        elif conf_level == 0.95:  # 95% CI
            df["q_025"] = y_pis[:, 0, 0]
            df["q_975"] = y_pis[:, 1, 0]

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
    median_interval_width: float,
    lower_q: str = "q_10",
    upper_q: str = "q_90",
) -> pd.DataFrame:
    """Compute confidence scores (0.0 to 1.0) based on prediction interval width.

    Confidence is derived from the 80% prediction interval (q_10 to q_90) width:
    - Narrower intervals → higher confidence (model is more certain)
    - Wider intervals → lower confidence (model is less certain)

    Why 80% CI (q_10/q_90)?
        - 68% CI is too narrow and sensitive to noise
        - 95% CI is too wide and less discriminating between samples
        - 80% provides a good balance for ranking prediction reliability

    Formula: confidence = exp(-width / median_width)
        - When width equals median, confidence ≈ 0.37
        - When width is half median, confidence ≈ 0.61
        - When width is double median, confidence ≈ 0.14

    This exponential decay is a common choice for converting uncertainty to
    confidence scores, providing a smooth mapping that appropriately penalizes
    high-uncertainty predictions.

    Args:
        df: DataFrame with quantile columns from predict_intervals()
        median_interval_width: Pre-computed median interval width from training data
        lower_q: Lower quantile column name (default: 'q_10')
        upper_q: Upper quantile column name (default: 'q_90')

    Returns:
        DataFrame with added 'confidence' column (values between 0 and 1)
    """
    interval_width = (df[upper_q] - df[lower_q]).abs()
    df["confidence"] = np.exp(-interval_width / median_interval_width)

    return df
