"""Classification UQ: VGMU + isotonic calibration.

For regression UQ, see workbench.algorithms.dataframe.uq_model_v1.UQModelV1
(proximity-augmented RandomForest error model) or
workbench.algorithms.dataframe.uq_model_v0.UQModelV0 (original isotonic-on-
(prediction, std) calibrator). The active version per model bundle is selected
by ``hyperparameters["uq_version"]`` (default ``"v0"``).

Classification approach:
    1. Each ensemble member outputs softmax probabilities
    2. VGMU (Variance-Gated Margin Uncertainty) combines the top-2 probability
       margin with ensemble disagreement on those classes via a signal-to-noise
       ratio:
            SNR = (p_top1 - p_top2) / (std_top1 + std_top2 + eps)
            gamma = 1 - exp(-SNR)
            raw_confidence = gamma * p_top1
    3. Isotonic regression maps raw_confidence → P(correct) on held-out data
    4. At inference, the calibrated mapping is applied via np.interp

Reference: Variance-Gated Ensembles (VGE), arXiv:2602.08142 (2025)

Usage:
    # Training:
    raw_conf = compute_vgmu_confidence(avg_probs, all_probs_stack)
    uq_metadata = calibrate_classification_confidence(raw_conf, y_true, y_pred)
    save_classification_uq(uq_metadata, model_dir)

    # Inference:
    uq_metadata = load_classification_uq(model_dir)
    raw_conf = compute_vgmu_confidence(avg_probs, all_probs_stack)
    confidence = apply_classification_confidence(raw_conf, uq_metadata["classification_confidence"])
"""

import json
import os
import numpy as np


def compute_vgmu_confidence(
    avg_probs: np.ndarray,
    all_probs_stack: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute raw VGMU (Variance-Gated Margin Uncertainty) confidence for classification.

    Combines the probability margin between top-2 classes with ensemble disagreement
    on those classes via a signal-to-noise ratio:

        SNR = (p_top1 - p_top2) / (std_top1 + std_top2 + eps)
        gamma = 1 - exp(-SNR)
        raw_confidence = gamma * p_top1

    Intuition:
        - High margin + low ensemble std → high SNR → gamma ≈ 1 → confidence ≈ p_top1
        - Low margin or high ensemble std → low SNR → gamma ≈ 0 → confidence ≈ 0
        - Single model (std=0) → gracefully degrades to p_top1 (max probability)
        - Uniform proba (margin=0) → confidence = 0

    Reference: Variance-Gated Ensembles (VGE), arXiv:2602.08142 (2025)

    Args:
        avg_probs (np.ndarray): Mean softmax probabilities, shape (n_samples, n_classes)
        all_probs_stack (np.ndarray): Per-model softmax probabilities,
            shape (n_models, n_samples, n_classes)
        eps (float): Small constant to prevent division by zero (default: 1e-8)

    Returns:
        np.ndarray: Raw confidence values, shape (n_samples,). NOT yet calibrated.
    """
    avg_probs = np.asarray(avg_probs)
    all_probs_stack = np.asarray(all_probs_stack)
    n = len(avg_probs)

    # Top-1 and top-2 class indices from averaged probabilities
    sorted_indices = np.argsort(-avg_probs, axis=1)  # descending
    top1_idx = sorted_indices[:, 0]
    top2_idx = sorted_indices[:, 1]

    # Mean probabilities for top-1 and top-2
    p_top1 = avg_probs[np.arange(n), top1_idx]
    p_top2 = avg_probs[np.arange(n), top2_idx]

    # Ensemble std for each class, then extract top-1 and top-2
    std_per_class = np.std(all_probs_stack, axis=0)  # (n_samples, n_classes)
    std_top1 = std_per_class[np.arange(n), top1_idx]
    std_top2 = std_per_class[np.arange(n), top2_idx]

    # VGMU formula
    snr = (p_top1 - p_top2) / (std_top1 + std_top2 + eps)
    gamma = 1.0 - np.exp(-snr)
    raw_confidence = gamma * p_top1

    return raw_confidence


def calibrate_classification_confidence(
    raw_confidence: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calibrate raw VGMU confidence to P(correct) using isotonic regression.

    Fits an isotonic (monotonically non-decreasing) mapping from raw confidence
    scores to empirical accuracy on validation data. The fitted mapping is stored
    as piecewise-linear thresholds for lightweight inference (just np.interp).

    Args:
        raw_confidence (np.ndarray): Raw VGMU confidence values, shape (n_samples,)
        y_true (np.ndarray): True labels (string or int), shape (n_samples,)
        y_pred (np.ndarray): Predicted labels (string or int), shape (n_samples,)

    Returns:
        dict: UQ metadata with "classification_confidence" key containing
            x_thresholds and y_thresholds for np.interp at inference time
    """
    from sklearn.isotonic import IsotonicRegression

    raw_confidence = np.asarray(raw_confidence).flatten()
    correctness = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)

    # Fit isotonic regression: raw_confidence → P(correct)
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(raw_confidence, correctness)

    # Extract piecewise-linear mapping for JSON serialization
    calibration_data = {
        "x_thresholds": iso_reg.X_thresholds_.tolist(),
        "y_thresholds": iso_reg.y_thresholds_.tolist(),
    }

    # Diagnostics
    calibrated = iso_reg.predict(raw_confidence)
    print("\n" + "=" * 50)
    print("Calibrating Classification Confidence (VGMU)")
    print("=" * 50)
    print(f"  Validation samples: {len(raw_confidence)}")
    print(f"  Overall accuracy: {correctness.mean():.3f}")
    print(f"  Raw confidence  - mean: {raw_confidence.mean():.3f}, std: {raw_confidence.std():.3f}")
    print(f"  Calibrated conf - mean: {calibrated.mean():.3f}, std: {calibrated.std():.3f}")

    # Reliability: bin by raw confidence, show actual accuracy per bin
    n_bins = 5
    bin_edges = np.percentile(raw_confidence, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10
    for i in range(n_bins):
        mask = (raw_confidence >= bin_edges[i]) & (raw_confidence < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = calibrated[mask].mean()
            print(f"  Bin {i + 1}: n={mask.sum():>5}, accuracy={bin_acc:.3f}, calibrated_conf={bin_conf:.3f}")

    return {"classification_confidence": calibration_data}


def apply_classification_confidence(
    raw_confidence: np.ndarray,
    calibration_data: dict,
) -> np.ndarray:
    """Apply saved isotonic calibration to raw VGMU confidence values.

    Uses np.interp for the piecewise-linear mapping — no sklearn needed at inference.

    Args:
        raw_confidence (np.ndarray): Raw VGMU confidence values, shape (n_samples,)
        calibration_data (dict): Dict with "x_thresholds" and "y_thresholds" arrays
            from calibrate_classification_confidence()

    Returns:
        np.ndarray: Calibrated confidence values in [0, 1], shape (n_samples,)
    """
    x_thresholds = np.array(calibration_data["x_thresholds"])
    y_thresholds = np.array(calibration_data["y_thresholds"])

    # Piecewise-linear interpolation (np.interp clamps to edge values for out-of-bounds)
    calibrated = np.interp(np.asarray(raw_confidence).flatten(), x_thresholds, y_thresholds)
    return np.clip(calibrated, 0.0, 1.0)


def save_classification_uq(uq_metadata: dict, model_dir: str) -> None:
    """Save classification UQ metadata to disk."""
    with open(os.path.join(model_dir, "classification_uq.json"), "w") as fp:
        json.dump(uq_metadata, fp, indent=2)
    print(f"Saved classification UQ metadata to {model_dir}")


def load_classification_uq(model_dir: str) -> dict:
    """Load classification UQ metadata from disk."""
    path = os.path.join(model_dir, "classification_uq.json")
    with open(path) as fp:
        return json.load(fp)
