"""Shared utilities for meta model aggregation.

Used by both the deployed template (meta_model.template) and the
MetaModelSimulator so that confidence and weight computations are
guaranteed to be identical.
"""

import numpy as np


def conf_weights_with_fallback(conf_arr: np.ndarray, fallback_w: np.ndarray) -> np.ndarray:
    """Compute normalized confidence weights, falling back to static weights for zero-confidence rows.

    Args:
        conf_arr: (N, M) array of confidence-based values (raw, scaled, or calibrated)
        fallback_w: (M,) array of static weights to use when row confidence sums to ~0

    Returns:
        (N, M) array of normalized per-row weights
    """
    conf_sum = conf_arr.sum(axis=1, keepdims=True)
    zero_conf = (conf_sum < 1e-12).ravel()
    return np.where(zero_conf[:, None], fallback_w, conf_arr / (conf_sum + 1e-12))


def ensemble_confidence(
    pred_arr: np.ndarray,
    conf_arr: np.ndarray,
    corr_scale: np.ndarray,
    model_weights: np.ndarray,
    optimal_alpha: float,
) -> np.ndarray:
    """Compute ensemble confidence by blending model agreement with calibrated confidence.

    confidence = alpha * agreement + (1 - alpha) * cal_conf

    where:
      - agreement = 1 / (1 + pred_std)  — high when models converge
      - cal_conf = (conf * corr_scale * model_weights).sum(axis=1)

    Args:
        pred_arr: (N, M) array of predictions from M models
        conf_arr: (N, M) array of confidences from M models
        corr_scale: (M,) array of |confidence-to-error correlation| per model
        model_weights: (M,) array of normalized model weights (sum to 1)
        optimal_alpha: Blend weight (0=calibrated conf only, 1=agreement only)

    Returns:
        (N,) array of ensemble confidence values
    """
    pred_std = pred_arr.std(axis=1)
    agreement = 1.0 / (1.0 + pred_std)
    cal_conf = (conf_arr * corr_scale * model_weights).sum(axis=1)
    return optimal_alpha * agreement + (1.0 - optimal_alpha) * cal_conf
