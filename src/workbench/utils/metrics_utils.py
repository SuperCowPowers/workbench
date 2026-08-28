"""Metrics utilities for computing model performance from predictions."""

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

log = logging.getLogger("workbench")


def reorder_cm_df(cm: pd.DataFrame, labels: List[str]) -> Optional[pd.DataFrame]:
    """Reorder a confusion-matrix DataFrame's rows and class-columns to the given label order.

    The input is expected to have a "labels" column plus one numeric column per class
    (the format produced by `generate_confusion_matrix`). Returns None if the label set
    in the DataFrame doesn't match `labels` (e.g., classes added or removed).

    Args:
        cm: Confusion matrix DataFrame with a "labels" column and per-class columns.
        labels: Desired class label order.

    Returns:
        Reordered DataFrame, or None if the label sets don't match.
    """
    if "labels" not in cm.columns:
        return None
    new_set = set(labels)
    row_set = set(cm["labels"])
    col_set = set(cm.columns) - {"labels"}
    if row_set != new_set or not new_set.issubset(col_set):
        return None
    return cm.set_index("labels").reindex(labels)[labels].reset_index()


def reorder_metrics_df(df: pd.DataFrame, labels: List[str]) -> Optional[pd.DataFrame]:
    """Reorder per-class rows in a metrics DataFrame to the given label order.

    Identifies the label column by finding the column whose values match `labels`
    (ignoring an optional 'all' aggregate row). Pins the 'all' row to the bottom
    if present. Returns None if no column matches the label set.

    Args:
        df: Metrics DataFrame (e.g., output of `compute_classification_metrics`).
        labels: Desired class label order.

    Returns:
        Reordered DataFrame, or None if no label column matches.
    """
    new_set = set(labels)
    label_col = None
    for col in df.columns:
        vals = set(df[col].astype(str)) - {"all"}
        if vals == new_set:
            label_col = col
            break
    if label_col is None:
        return None

    all_mask = df[label_col].astype(str) == "all"
    all_rows = df[all_mask]
    class_rows = df[~all_mask].set_index(label_col).reindex(labels).reset_index()
    return pd.concat([class_rows, all_rows], ignore_index=True)


def validate_proba_columns(predictions_df: pd.DataFrame, class_labels: List[str], guessing: bool = False) -> bool:
    """Validate that probability columns match class labels.

    Args:
        predictions_df: DataFrame with prediction results
        class_labels: List of class labels
        guessing: Whether class labels were guessed from data

    Returns:
        True if validation passes

    Raises:
        ValueError: If probability columns don't match class labels
    """
    proba_columns = [col.replace("_proba", "") for col in predictions_df.columns if col.endswith("_proba")]

    if sorted(class_labels) != sorted(proba_columns):
        label_type = "GUESSED class_labels" if guessing else "class_labels"
        raise ValueError(f"_proba columns {proba_columns} != {label_type} {class_labels}!")
    return True


def compute_classification_metrics(
    predictions_df: pd.DataFrame,
    target_col: str,
    class_labels: Optional[List[str]] = None,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute classification metrics from a predictions DataFrame.

    Args:
        predictions_df: DataFrame with target and prediction columns
        target_col: Name of the target column
        class_labels: List of class labels in order (if None, inferred from target column)
        prediction_col: Name of the prediction column (default: "prediction")

    Returns:
        DataFrame with per-class metrics (precision, recall, f1, roc_auc, support)
        plus a weighted 'all' row. Returns empty DataFrame if validation fails.
    """
    # Validate inputs
    if predictions_df.empty:
        log.warning("Empty DataFrame provided. Returning empty metrics.")
        return pd.DataFrame()

    if prediction_col not in predictions_df.columns:
        log.warning(f"Prediction column '{prediction_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    if target_col not in predictions_df.columns:
        log.warning(f"Target column '{target_col}' not found in DataFrame. Returning empty metrics.")
        return pd.DataFrame()

    # Handle NaN predictions
    df = predictions_df.copy()
    nan_pred = df[prediction_col].isnull().sum()
    if nan_pred > 0:
        log.warning(f"Dropping {nan_pred} rows with NaN predictions.")
        df = df[~df[prediction_col].isnull()]

    if df.empty:
        log.warning("No valid rows after dropping NaNs. Returning empty metrics.")
        return pd.DataFrame()

    # Handle class labels
    guessing = False
    if class_labels is None:
        log.warning("Class labels not provided. Inferring from target column.")
        class_labels = df[target_col].unique().tolist()
        guessing = True

    # Validate probability columns if present
    proba_cols = [col for col in df.columns if col.endswith("_proba")]
    if proba_cols:
        validate_proba_columns(df, class_labels, guessing=guessing)

    y_true = df[target_col]
    y_pred = df[prediction_col]

    # Precision, recall, f1, support per class
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_labels, zero_division=0)

    # ROC AUC per class (requires probability columns and sorted labels)
    proba_col_names = [f"{label}_proba" for label in class_labels]
    if all(col in df.columns for col in proba_col_names):
        # roc_auc_score requires labels to be sorted, so we sort and reorder results back
        sorted_labels = sorted(class_labels)
        sorted_proba_cols = [f"{label}_proba" for label in sorted_labels]
        y_score_sorted = df[sorted_proba_cols].values
        roc_auc_sorted = roc_auc_score(y_true, y_score_sorted, labels=sorted_labels, multi_class="ovr", average=None)
        # Map back to original class_labels order
        label_to_auc = dict(zip(sorted_labels, roc_auc_sorted))
        roc_auc = np.array([label_to_auc[label] for label in class_labels])
    else:
        roc_auc = np.array([None] * len(class_labels))

    # Build per-class metrics
    metrics_df = pd.DataFrame(
        {
            target_col: class_labels,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "support": support.astype(int),
        }
    )

    # Add weighted 'all' row
    total = support.sum()
    all_row = {
        target_col: "all",
        "precision": (prec * support).sum() / total,
        "recall": (rec * support).sum() / total,
        "f1": (f1 * support).sum() / total,
        "roc_auc": (roc_auc * support).sum() / total if roc_auc[0] is not None else None,
        "support": int(total),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([all_row])], ignore_index=True)

    return metrics_df


# The metric names that reach SageMaker training-job telemetry. `print_regression_metrics`
# emits exactly these and `FeaturesToModel` builds its metric_definitions regexes from the
# same list, so the two halves of that stdout contract cannot drift apart in source. They
# still deploy on different clocks: the regexes are client-side, the prints are baked into
# the image at WORKBENCH_VERSION. Ship the image first when adding, the regex first when removing.
TRAINING_METRICS = ("rmse", "mae", "r2", "spearmanr", "support")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics from aligned truth/prediction arrays.

    `pearsonr` squared is the best R2 any affine rescale of the predictions can reach, so
    the distance from `r2` to it is placement rather than ranking. Where that distance sits
    — a wrong centre, or a spread that does not match the correlation — is measurable off a
    predictions frame, so it is left to the caller rather than carried on every row.

    Args:
        y_true: Ground truth target values
        y_pred: Predicted values

    Returns:
        Dict of metric name -> value (rmse, mae, r2, pearsonr, spearmanr, support).
        Empty dict if no pair survives NaN removal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    keep = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not keep.all():
        log.warning(f"NaNs found: dropping {int((~keep).sum())} of {len(keep)} rows.")
        y_true, y_pred = y_true[keep], y_pred[keep]

    if len(y_true) == 0:
        log.warning("No valid rows after dropping NaNs. Returning empty metrics.")
        return {}

    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearsonr": pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan,
        "spearmanr": spearmanr(y_true, y_pred).correlation,
        "support": len(y_true),
    }


def print_regression_metrics(metrics: Dict[str, float]) -> None:
    """Print `TRAINING_METRICS` in the format SageMaker's metric definitions scrape.

    Args:
        metrics: Output of `compute_regression_metrics`

    Raises:
        ValueError: If `metrics` is empty -- every truth/prediction pair was NaN.
    """
    if not metrics:
        raise ValueError("No regression metrics to print: every truth/prediction pair was NaN.")
    for name in TRAINING_METRICS:
        value = metrics[name]
        print(f"{name}: {value}" if isinstance(value, int) else f"{name}: {value:.3f}")


def soft_threshold_error(
    y_pred: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
) -> np.ndarray:
    """Per-sample distance from a prediction to a credible interval.

    Predictions inside the interval score zero; outside, the error is the distance
    to the nearest bound. For targets fit as Bayesian dose-response curves, this
    scores against the measurement's own uncertainty instead of a point estimate.

    Args:
        y_pred: Predicted values
        ci_lower: Lower bound of each label's credible interval
        ci_upper: Upper bound of each label's credible interval

    Returns:
        Array of per-sample errors, zero wherever the prediction lands inside the interval.
    """
    y_pred, ci_lower, ci_upper = np.asarray(y_pred), np.asarray(ci_lower), np.asarray(ci_upper)
    return np.maximum(0.0, np.maximum(ci_lower - y_pred, y_pred - ci_upper))


def soft_threshold_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    soft_baseline: bool = False,
) -> float:
    """Soft-Threshold Relative Absolute Error (ST-RAE). Lower is better.

    Relative absolute error where per-sample error is `soft_threshold_error`: the
    distance outside the label's credible interval, zero inside. The denominator is
    the mean-predictor baseline's error, matching the `rae` published in OpenADMET's
    challenge tutorial, which originated the metric.

    Args:
        y_true: Ground-truth values (the curve-fit point estimates)
        y_pred: Predicted values
        ci_lower: Lower bound of each label's credible interval
        ci_upper: Upper bound of each label's credible interval
        soft_baseline: Score the baseline through the same soft threshold, so 1.0 keeps
            the ordinary-RAE reading of "no better than predicting the mean". Inflates
            the score against the published form by a factor that varies per target
            (0.49-0.68 on CYP), so the two are not interconvertible.

    Returns:
        ST-RAE, or NaN when the baseline error is zero (no signal to normalize against).
    """
    y_true = np.asarray(y_true)
    numerator = soft_threshold_error(y_pred, ci_lower, ci_upper).sum()

    baseline = np.full_like(y_true, y_true.mean(), dtype=float)
    if soft_baseline:
        denominator = soft_threshold_error(baseline, ci_lower, ci_upper).sum()
    else:
        denominator = np.abs(y_true - baseline).sum()

    if denominator == 0:
        log.warning("Baseline error is zero, ST-RAE is undefined. Returning NaN.")
        return float("nan")
    return float(numerator / denominator)


def bootstrap_metric(
    predictions_df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Point estimate and bootstrap spread for any metric over a predictions frame.

    Resamples rows with replacement and recomputes the metric, which turns a single
    score into a score plus the uncertainty attached to it. A difference between two
    models smaller than this spread is not a result.

    Prefer `bootstrap_compare` when the question is which of two models is better:
    two overlapping intervals from this function do not mean the models are
    indistinguishable, since it cannot cancel out per-row difficulty.

    Args:
        predictions_df: Predictions frame, one row per scored unit
        metric_fn: Takes a frame shaped like `predictions_df`, returns a scalar score
        n_resamples: Bootstrap resamples to draw
        seed: Fixed so repeated calls on the same data agree

    Returns:
        Dict with "value" (the point estimate), "std", "ci_lower", "ci_upper" (2.5th
        and 97.5th percentiles), and "n" (rows scored).
    """
    n = len(predictions_df)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [metric_fn(predictions_df.iloc[rng.choice(n, n, replace=True)]) for _ in range(n_resamples)], dtype=float
    )
    lower, upper = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "value": float(metric_fn(predictions_df)),
        "std": float(np.nanstd(draws)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n": n,
    }


def bootstrap_compare(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_resamples: int = 1000,
    seed: int = 0,
    lower_is_better: bool = True,
) -> Dict[str, float]:
    """Paired bootstrap comparing two models scored on the same rows.

    Both frames are resampled on the *same* drawn rows, so per-row difficulty cancels
    and only the difference between the models is left. That is far more sensitive
    than comparing each model's own interval: two models can have heavily overlapping
    marginal intervals while one beats the other on nearly every resample.

    Frames are paired on their index, so index both by the same identifier (e.g. the
    id column). Rows missing from either frame are dropped.

    Args:
        predictions_a: Predictions from the first model, indexed by row id
        predictions_b: Predictions from the second model, indexed by row id
        metric_fn: Takes a frame shaped like either input, returns a scalar score
        n_resamples: Bootstrap resamples to draw
        seed: Fixed so repeated calls on the same data agree
        lower_is_better: True for error metrics (ST-RAE, MAE), False for R2, MCC and
            other higher-is-better scores. Sets which sign of `delta` counts as a win.

    Returns:
        Dict with "delta" (a minus b), "ci_lower"/"ci_upper" for that difference,
        "p_a_better" (fraction of resamples where a wins), the two point estimates
        "value_a"/"value_b", and "n" (paired rows). A "ci_lower" to "ci_upper" range
        spanning zero means the comparison did not separate the models.
    """
    shared = predictions_a.index.intersection(predictions_b.index)
    dropped = (len(predictions_a) - len(shared)) + (len(predictions_b) - len(shared))
    if dropped:
        log.warning(f"bootstrap_compare pairing on {len(shared)} shared rows, dropped {dropped} unpaired")
    a, b = predictions_a.loc[shared], predictions_b.loc[shared]

    n = len(shared)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        rows = rng.choice(n, n, replace=True)
        deltas[i] = metric_fn(a.iloc[rows]) - metric_fn(b.iloc[rows])

    lower, upper = np.nanpercentile(deltas, [2.5, 97.5])
    wins = (deltas < 0) if lower_is_better else (deltas > 0)
    return {
        "delta": float(metric_fn(a) - metric_fn(b)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_a_better": float(np.nanmean(wins)),
        "value_a": float(metric_fn(a)),
        "value_b": float(metric_fn(b)),
        "n": n,
    }


def resolve_primary_target(targets: Union[str, List[str], None]) -> Optional[str]:
    """The target a model is scored on.

    Multi-task models carry a list of targets and are scored on the first.

    Args:
        targets: A single target column, a list of them, or None

    Returns:
        The target column to score on, or None
    """
    if isinstance(targets, list):
        return targets[0] if targets else None
    return targets


def default_inference_run(inference_runs: List[str]) -> Optional[str]:
    """Pick the inference run to report when the caller didn't name one.

    Priority: full_cross_fold -> test_inference -> first available. The
    model_training capture is excluded, since it reports the training job's own
    metrics rather than an inference pass.

    Args:
        inference_runs: The available run names

    Returns:
        The run name, or None if there are none to pick from
    """
    runs = [run for run in inference_runs if run != "model_training"]
    for preferred in ("full_cross_fold", "test_inference"):
        if preferred in runs:
            return preferred
    return runs[0] if runs else None


def compute_metrics_from_predictions(
    predictions_df: pd.DataFrame,
    target_col: str,
    class_labels: Optional[List[str]] = None,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Compute metrics from a predictions DataFrame.

    Automatically determines if this is classification or regression based on
    whether class_labels is provided.

    Args:
        predictions_df: DataFrame with target and prediction columns
        target_col: Name of the target column
        class_labels: List of class labels for classification (None for regression)
        prediction_col: Name of the prediction column (default: "prediction")

    Returns:
        DataFrame with computed metrics
    """
    if target_col not in predictions_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in predictions DataFrame")
    if prediction_col not in predictions_df.columns:
        raise ValueError(f"Prediction column '{prediction_col}' not found in predictions DataFrame")

    if class_labels:
        return compute_classification_metrics(predictions_df, target_col, class_labels, prediction_col)
    metrics = compute_regression_metrics(predictions_df[target_col], predictions_df[prediction_col])
    return pd.DataFrame([metrics]) if metrics else pd.DataFrame()


if __name__ == "__main__":
    # Test with sample data
    print("Testing classification metrics...")
    class_df = pd.DataFrame(
        {
            "target": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "prediction": ["a", "b", "c", "a", "b", "a", "a", "b", "c", "b"],
            "a_proba": [0.8, 0.1, 0.1, 0.7, 0.2, 0.4, 0.9, 0.1, 0.1, 0.3],
            "b_proba": [0.1, 0.8, 0.1, 0.2, 0.7, 0.3, 0.05, 0.8, 0.2, 0.6],
            "c_proba": [0.1, 0.1, 0.8, 0.1, 0.1, 0.3, 0.05, 0.1, 0.7, 0.1],
        }
    )
    metrics = compute_metrics_from_predictions(class_df, "target", ["a", "b", "c"])
    print(metrics.to_string(index=False))

    print("\nTesting regression metrics...")
    reg_df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "prediction": [1.1, 2.2, 2.9, 4.1, 4.8],
        }
    )
    metrics = compute_metrics_from_predictions(reg_df, "target")
    print(metrics.to_string(index=False))


def placement_metrics(
    predictions_df: pd.DataFrame,
    target_col: str,
    prediction_col: str = "prediction",
) -> Dict[str, float]:
    """Where predictions sit relative to the targets, independent of how well they rank.

    `r2_opt` is `pearsonr` squared: the best R2 reachable by rescaling the predictions,
    leaving their order untouched. `gap` is `r2_opt - r2`, the part of the loss that is
    placement rather than ranking, and it splits exactly into
    `(pearsonr - spread_ratio)**2 + (bias/sd(true))**2` — so the other two say which half
    to fix. `spread_ratio` is sd(pred)/sd(true), and its target is `pearsonr`, not 1:
    squared loss shrinks predictions toward the mean on purpose, so well under `pearsonr`
    is the signal rather than under 1. `bias` is mean(pred) - mean(true) in target units.

    Only meaningful when the targets come from the deployment population; on a random
    split `bias` is 0 by construction. Treat `gap` as a detector, not an estimator — its
    bootstrap interval is wide at low support, so it says whether to look, not how much
    to correct.

    Args:
        predictions_df: DataFrame from `Model.get_inference_predictions()`
        target_col: Name of the target column
        prediction_col: Prediction column. For a multi-target model pass
            f"{target_col}_pred" — the bare "prediction" column holds the first target.

    Returns:
        Dict with "r2", "r2_opt", "gap", "spread_ratio", "bias". Empty dict if no pair
        survives NaN removal.
    """
    for col in (target_col, prediction_col):
        if col not in predictions_df.columns:
            raise ValueError(f"Column '{col}' not found in predictions DataFrame")

    pair = predictions_df[[target_col, prediction_col]].dropna()
    if pair.empty:
        log.warning("No valid rows after dropping NaNs. Returning empty metrics.")
        return {}

    y_true = pair[target_col].to_numpy(dtype=float)
    y_pred = pair[prediction_col].to_numpy(dtype=float)
    sd_true = y_true.std()
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    r2_opt = pearson**2 if np.isfinite(pearson) else np.nan
    return {
        "r2": r2,
        "r2_opt": r2_opt,
        # Theorem, so the clamp only absorbs float noise: r2 <= pearsonr**2 for any predictor.
        "gap": max(0.0, r2_opt - r2) if np.isfinite(r2_opt) else np.nan,
        "spread_ratio": y_pred.std() / sd_true if sd_true > 0 else np.nan,
        "bias": y_pred.mean() - y_true.mean(),
    }
