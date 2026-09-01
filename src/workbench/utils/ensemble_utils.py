"""Shared utilities for ensemble aggregation math.

Used by :mod:`workbench.utils.aggregation_nodes` (the prediction
aggregator subclasses) and :class:`workbench.utils.ensemble_simulator.EnsembleSimulator`
so that the confidence/weight computations stay identical between
DAG-runtime aggregation and offline strategy simulation.
"""

import logging

import numpy as np

log = logging.getLogger("workbench")


def conf_weights_with_fallback(conf_arr: np.ndarray, fallback_w: np.ndarray) -> np.ndarray:
    """Compute normalized confidence weights, falling back to static weights per row.

    A row falls back when its confidences carry no usable signal: they sum to ~0, or any
    member's is missing. Missing counts because a model with no confidence cannot be
    weighted against one that has it — zeroing it would silently drop that model from the
    average, which is a larger change than weighting the row statically.

    Every returned row is finite. That matters: this feeds both the deployed aggregation
    nodes and the offline simulator, so a non-finite weight becomes a NaN *prediction* —
    served in one path, and in the other silently skipped by a NaN-tolerant mean, which
    scores that strategy on a different row set than its rivals.

    Args:
        conf_arr: (N, M) array of confidence-based values (raw, scaled, or calibrated)
        fallback_w: (M,) array of static weights to use where confidence is unusable

    Returns:
        (N, M) array of normalized per-row weights
    """
    conf = np.asarray(conf_arr, dtype=float)
    usable = np.isfinite(conf).all(axis=1)
    conf_sum = np.where(usable, conf.sum(axis=1), 0.0)
    fallback = ~usable | (conf_sum < 1e-12)

    n_fallback = int(fallback.sum())
    if n_fallback:
        n_missing = int((~usable).sum())
        log.warning(
            f"conf_weights_with_fallback: {n_fallback:,} of {len(conf):,} rows using static weights "
            f"({n_missing:,} with a missing confidence). Confidence weighting is inactive for those rows."
        )

    # Zero the fallback rows before dividing so no NaN reaches the arithmetic.
    safe = np.where(fallback[:, None], 0.0, conf)
    return np.where(fallback[:, None], fallback_w, safe / (conf_sum[:, None] + 1e-12))


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
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    conf_arr = np.asarray(conf_arr, dtype=np.float64)
    pred_std = pred_arr.std(axis=1)
    agreement = 1.0 / (1.0 + pred_std)
    cal_conf = (conf_arr * corr_scale * model_weights).sum(axis=1)
    return optimal_alpha * agreement + (1.0 - optimal_alpha) * cal_conf
