"""Shared XGBoost ensemble-member training, used by the template and the HPO objective.

One function builds and fits one fold's estimator, so a searched config maps to the same
model the template publishes — the property that makes an HPO result mean anything. The
template's publish loop and :mod:`workbench.training.xgb_hpo`'s trial function both call
:func:`train_xgb_fold`.

Early stopping watches a slice carved out of the fold's *training* rows, never the fold's
validation rows. Those validation rows become out-of-fold predictions, and the regression
UQ error model is fit on their residuals; stopping at whichever iteration looked best on
them would make that error model learn to under-predict uncertainty.

Training-only; imported from inside a model template's ``__main__``.
"""

from __future__ import annotations

# Workbench knobs the template consumes itself — never forwarded to XGBoost.
WORKBENCH_PARAMS = {"n_folds", "split_strategy", "butina_cutoff", "uq_version", "early_stopping_fraction"}


def align_frame(df, category_mappings=None, orig_features=None, compressed_features=None):
    """Route a frame through the template's fitted preprocessing.

    The held-out validation rows are split off *before* the template fits its
    preprocessing, so they arrive raw; this applies the same fitted transforms —
    categorical mappings and compressed-feature decompression. Idempotent, so it is
    safe on a frame that already went through the template's own preprocessing.

    Args:
        df: the frame to align (not mutated).
        category_mappings: the template's fitted categorical mappings.
        orig_features: the feature list before decompression.
        compressed_features: features stored as bitstrings/count vectors.

    Returns:
        The aligned copy.
    """
    from workbench.endpoints.inference import convert_categorical_types, decompress_features

    df = df.copy()
    # An empty frame (no validation_ids) has nothing to transform, and decompression
    # rejects a 0-row column.
    if df.empty:
        return df
    if category_mappings:
        df, _ = convert_categorical_types(df, list(category_mappings), category_mappings)
    if compressed_features and any(f in df.columns for f in compressed_features):
        df, _ = decompress_features(df, orig_features, compressed_features)
    return df


def xgb_params(hyperparameters: dict, *, fold_idx: int = 0) -> dict:
    """Reduce a hyperparameters dict to the estimator kwargs XGBoost accepts.

    Workbench-only knobs are dropped, ``seed`` becomes ``random_state``, regression defaults
    to MAE, and each fold offsets the seed so the ensemble members differ.
    """
    params = {k: v for k, v in hyperparameters.items() if k not in WORKBENCH_PARAMS and k != "hpo"}
    params["random_state"] = params.pop("seed", 42) + fold_idx
    params.setdefault("objective", "reg:absoluteerror")
    return params


def train_xgb_fold(
    hyperparameters: dict,
    X,
    y,
    *,
    sample_weight=None,
    model_type: str = "regressor",
    fold_idx: int = 0,
):
    """Fit one ensemble member on ``X``/``y`` and return it.

    When ``early_stopping_rounds`` is set, ``early_stopping_fraction`` of these rows are held
    back as the watch set and the rest do the fitting; the returned estimator predicts at its
    own ``best_iteration``. Without it, every row fits and ``n_estimators`` is taken literally.

    Args:
        hyperparameters: merged hyperparameters (Workbench knobs are filtered out here).
        X: fold training features.
        y: fold training targets.
        sample_weight: per-row weights aligned with ``X``, or None.
        model_type: ``"classifier"`` or ``"regressor"``.
        fold_idx: ensemble member index — offsets the seed so members differ.

    Returns:
        The fitted XGBoost estimator.
    """
    import numpy as np
    import xgboost as xgb

    params = xgb_params(hyperparameters, fold_idx=fold_idx)
    if model_type == "classifier":
        params.pop("objective", None)  # regression-only default
    builder = xgb.XGBClassifier if model_type == "classifier" else xgb.XGBRegressor
    model = builder(enable_categorical=True, **params)

    if not params.get("early_stopping_rounds"):
        model.fit(X, y, sample_weight=sample_weight)
        return model

    # Carve the watch set out of the fold's own training rows, seeded off the member's seed
    # so a re-run reproduces the split.
    fraction = hyperparameters.get("early_stopping_fraction", 0.1)
    order = np.random.default_rng(params["random_state"]).permutation(len(X))
    n_watch = min(len(X) - 1, max(1, int(round(len(X) * fraction))))
    watch, fit = order[:n_watch], order[n_watch:]

    model.fit(
        X.iloc[fit],
        y[fit] if isinstance(y, np.ndarray) else y.iloc[fit],
        sample_weight=None if sample_weight is None else np.asarray(sample_weight)[fit],
        eval_set=[(X.iloc[watch], y[watch] if isinstance(y, np.ndarray) else y.iloc[watch])],
        verbose=False,
    )
    return model
