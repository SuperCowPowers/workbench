"""XGBoost hyperparameter-search objective + default search space.

The XGBoost adapter for :mod:`workbench.training.hpo_runner`: a default search space
(this module) and, per trial, a full ``n_folds`` ensemble scored as ``holdout_mae`` /
``cv_mae`` (the search *objective*). Scoring a trial in the regime the winner is
published in is the point — a config selected as a lone model does not carry over to an
ensemble. The runner owns everything around that: selection, the re-rank, the artifacts.

Trials are cheap here — one fit is seconds, not minutes — so a wider search space and a
larger trial budget are affordable in a way they are not for chemprop.

Serial by design: a single ``XGBRegressor.fit`` already spreads across every core, so
concurrent trials would contend for the same CPUs rather than add throughput. The base
training image carries optuna alone for that reason; ``backend="ray"`` needs an image
that has it.

Training-only; imported **only inside the XGBoost template's ``__main__``** (deferred).
"""

from __future__ import annotations

import os

from workbench.training.hpo_harness import FloatRange, IntRange
from workbench.training.hpo_runner import HpoAdapter, run_hpo

# Workbench knobs the template consumes itself — never forwarded to XGBoost.
# Default per-knob search space, split into the two levers that matter for gradient
# boosting. Both groups are searched by default. Everything else — n_folds,
# split_strategy, uq_version, objective, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    # Capacity and how fast it is spent. n_estimators is NOT here — early stopping picks the
    # round count per fold, so the tree budget tunes itself and a searched value would only
    # fight it. max_depth and min_child_weight set how expressive an individual tree is.
    "basic": {
        "max_depth": IntRange(3, 16, 1, default=7),
        "min_child_weight": IntRange(1, 30, 1, default=3),
        "learning_rate": FloatRange(0.003, 0.3, log=True, default=0.05),
    },
    # Sampling + penalty terms. These are the knobs that decide how hard the model
    # overfits, which is the dominant failure mode on the small, wide descriptor frames
    # ADMET datasets produce. gamma starts at 0 (no split penalty), so it is sampled
    # linearly rather than log. The reg_alpha/reg_lambda floors sit far enough below
    # XGBoost's own defaults to be numerically off — a log range cannot include 0, and
    # searches on ADMET data head for that corner.
    "reg": {
        "subsample": FloatRange(0.4, 1.0, step=0.05, default=0.8),
        "colsample_bytree": FloatRange(0.4, 1.0, step=0.05, default=0.8),
        "gamma": FloatRange(0.0, 5.0, step=0.1, default=0.1),
        "reg_alpha": FloatRange(1e-4, 10.0, log=True, default=0.1),
        "reg_lambda": FloatRange(1e-3, 50.0, log=True, default=1.0),
    },
}


def xgb_search_space(groups=("basic", "reg")) -> dict:
    """Build the default XGBoost search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (capacity + boosting rate) and/or
            ``"reg"`` (sampling + regularization). Both by default.

    Returns:
        dict: ``{knob: Spec}`` for :func:`workbench.training.hpo_harness.run_search`.
    """
    space = {}
    for group in groups:
        if group not in _SEARCH_GROUPS:
            raise ValueError(f"Unknown search group {group!r}. Known: {sorted(_SEARCH_GROUPS)}")
        space.update(_SEARCH_GROUPS[group])
    return space


def resolve_search_space(spec) -> dict:
    """Resolve an ``hpo['search_space']`` value into a ``{knob: Spec}`` space.

    Accepts a shorthand string (``"basic"``, ``"basic+reg"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to all groups (``basic+reg``).
    """
    if spec is None:
        return xgb_search_space()
    if isinstance(spec, str):
        return xgb_search_space(spec.split("+"))
    if isinstance(spec, dict):
        return spec
    return xgb_search_space(tuple(spec))


class XGBAdapter(HpoAdapter):
    """Trains and scores one XGBoost candidate for :func:`run_hpo`.

    Regression only — the objective is MAE, and the runner's re-rank compares on it.
    """

    def __init__(self, *, target: str, features: list):
        self.target = target
        self.features = list(features)

    def resources_per_trial(self, hpo_block, backend):
        """Ray only. One trial claims the cores its estimator is configured to use."""
        if backend == "optuna":
            return None
        return {"cpu": _xgb_threads(max(1, hpo_block.get("max_parallel", 1)))}

    def make_trial_fn(self, *, train_df, folds, val_df, hyperparameters, metric, concurrency):
        """Build the ensemble XGBoost ``trial_fn`` (closes over the folds and eval data).

        Each trial trains one regressor per fold the way the template publishes them —
        same estimator kwargs, same per-fold seed offset, same sample weights — and scores
        the ensemble the way it is deployed:

        * ``holdout_mae`` — every member predicts ``val_df``; the objective is the MAE of the
          members' mean prediction, i.e. the real ensemble's held-out error.
        * ``cv_mae`` — each member is scored on its own out-of-fold rows and the objective is
          the mean across folds.
        """
        import numpy as np

        from workbench.training.xgb_core import train_xgb_fold

        n_jobs = _xgb_threads(concurrency)
        print(f"[hpo] {n_jobs} XGBoost threads per trial at {concurrency} concurrent")

        holdout = bool(len(val_df))
        all_y = train_df[self.target].to_numpy(dtype=float)
        eval_y = val_df[self.target].to_numpy(dtype=float) if holdout else None
        weights = train_df["sample_weight"] if "sample_weight" in train_df.columns else None

        def trial_fn(config, report):
            merged = {**self.merge_config(hyperparameters, config), "n_jobs": n_jobs}

            member_preds, fold_maes = [], []
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                model = train_xgb_fold(
                    merged,
                    train_df.iloc[tr_idx][self.features],
                    all_y[tr_idx],
                    sample_weight=None if weights is None else weights.iloc[tr_idx].to_numpy(),
                    fold_idx=fold_idx,
                )

                if holdout:
                    # Each member predicts the held-out rows; the objective is the ensemble's
                    # mean prediction, so the objective set never influences model selection.
                    member_preds.append(model.predict(val_df[self.features]))
                    running = float(np.nanmean(np.abs(np.mean(member_preds, axis=0) - eval_y)))
                else:
                    fold_preds = model.predict(train_df.iloc[va_idx][self.features])
                    fold_maes.append(float(np.nanmean(np.abs(fold_preds - all_y[va_idx]))))
                    running = float(np.mean(fold_maes))

                # Fold-granular pruning: a config already off the pace stops here rather than
                # paying for the remaining members.
                report(step=fold_idx + 1, **{metric: running})

            return running

        return trial_fn


def run_xgb_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    target: str,
    features: list,
    smiles_column: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the XGBoost hyperparameter search; returns the phase-2 hyperparameters.

    See :func:`workbench.training.hpo_runner.run_hpo` for the search/re-rank contract and
    the ``hpo`` block's keys.
    """
    return run_hpo(
        train_df,
        val_df,
        base_hyperparameters,
        hpo_block,
        adapter=XGBAdapter(target=target, features=features),
        search_space=resolve_search_space(hpo_block.get("search_space")),
        primary_target=target,
        smiles_column=smiles_column,
        output_dir=output_dir,
    )


def _xgb_threads(concurrency: int) -> int:
    """Estimator threads per trial at a given concurrency.

    One fit spreads across every core it is given, so concurrent trials have to divide them
    or they contend. Phase 1.5 runs fewer trials at once than the search, so it recomputes
    this rather than inheriting the search's tighter budget.
    """
    return max(1, (os.cpu_count() or 4) // max(1, concurrency))
