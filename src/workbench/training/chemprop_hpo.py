"""Chemprop hyperparameter-search objective + default search space.

Drives the framework-agnostic :mod:`workbench.training.hpo_harness` for chemprop: a
default search space (this module) and, per trial, a full ``n_folds`` ensemble scored as
``holdout_mae`` / ``cv_mae`` (the search *objective*). Scoring a trial in the regime the
winner is published in is the point — a config selected as a lone model does not carry
over to an ensemble. Training-only; imported **only inside the chemprop template's
``__main__``** (deferred).

Parity: each trial builds its model through
:func:`workbench.training.chemprop_core.build_mpnn_model` — the same builder the
template uses to publish the winner — so a searched config maps to the identical
architecture and optimizer schedule.

Only the pure pieces here (the search space and config merge) import without
chemprop; the search entry point and per-trial train defer their chemprop/chemprop_core
imports so this module stays importable for unit tests.
"""

from __future__ import annotations

import os

from workbench.training.hpo_harness import Choice, FloatRange, IntRange

# Trials report once per completed fold, so a trial is eligible for pruning only after its
# second member has trained. One fold is too noisy a basis to kill a config on: a config can
# lag on a single scaffold fold and still make the better ensemble.
FOLD_PRUNE_WARMUP = 2

# Default per-knob search space, grouped like chemprop's own hpopt keywords. The `basic`
# group is chemprop's canonical space verbatim, extended only so ffn_hidden_dim can also
# express a tapered head. Everything else — uq_version, max_epochs, patience, batch_size,
# split_strategy, criterion, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    "basic": {
        # depth / hidden_dim / ffn_num_layers ranges are chemprop's canonical search space
        # verbatim. ffn_hidden_dim additionally offers tapered heads (lists), which
        # chemprop's own space can't express — ffn_num_layers is ignored when a tapered
        # list is chosen (the list length sets the depth).
        "depth": IntRange(2, 6, 1),
        "hidden_dim": IntRange(300, 2400, 100),
        "ffn_num_layers": IntRange(1, 3, 1),
        "ffn_hidden_dim": Choice([300, 600, 1200, 1800, 2400, [1024, 256, 64], [512, 128]]),
    },
    # Opt-in ("basic+lr"). init_lr/final_lr are tied to max_lr in merge_best_config
    # rather than searched independently (independent search can produce init > max,
    # which the Noam schedule rejects).
    "lr": {
        "max_lr": FloatRange(1e-4, 5e-3, log=True),
        "warmup_epochs": IntRange(2, 10, 2),
    },
}

# Knobs outside the default space — the working list for when this is revisited, ordered by
# expected payoff. The full cross-file backlog (infra + correctness items too) lives in the
# "Backlog / follow-ups" section of docs/planning/hpo_support.md; this comment covers only
# the search-space knobs. Capacity search (the current `basic` group) is a modest lever:
# Yang et al. 2019 measured ~2-5% from HPO on most datasets, and on AqSol the search drove
# capacity to the floor for a small win. The bigger levers are featurization and LR/batch —
# and ensembling, which the n_folds publish already captures.
#
#   * featurization (atom-featurizer mode ORGANIC; RIGR for small datasets) — the biggest
#     untapped lever and not even in chemprop's "all" keyword. A template featurizer swap,
#     not a search range, so it lands as its own feature rather than a knob here.
#   * `batch_size` + the learning-rate group (`max_lr`, `warmup_epochs`, already the opt-in
#     "lr" group) — what chemprop's maintainers actually recommend tuning. Cheapest
#     high-value promotion into the default; batch_size also raises GPU utilization (trials
#     use ~11% of L4 memory at bs=64). batch_size range must be chosen against the training
#     instance's GPU memory, not the literature.
#   * capacity floor — the AqSol optimum sat on `hidden_dim=300` (the range floor), so lower
#     the floor before re-running to see if it wants to go smaller still.
#   * `dropout` — chemprop searches it ({0.0,0.05,…,0.4}) and ensemble-scored trials can now
#     select it honestly (a regularization knob is only meaningful against the ensemble it
#     ships in). Low priority — coupled to the ensemble, small effect.
#   * `aggregation` (mean/sum/norm) — searchable in chemprop but unreachable from here:
#     `chemprop_core.build_mpnn_model` constructs `NormAggregation` directly, so tuning it
#     requires a model-construction change first.
#   * `activation`, `aggregation_norm` — the remainder of chemprop's "all" keyword; smallest
#     expected effect.
#
# Each added knob costs trials: the default space is already 4-dimensional, pruning reserves
# the first PRUNE_STARTUP_TRIALS trials as un-pruned baselines, and every trial now trains a
# full ensemble.


def chemprop_search_space(groups=("basic",)) -> dict:
    """Build the default chemprop search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture capacity) and/or
            ``"lr"`` (the learning-rate schedule).

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

    Accepts a shorthand string (``"basic"``, ``"basic+lr"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to the ``basic`` group.
    """
    if spec is None:
        return chemprop_search_space()
    if isinstance(spec, str):
        return chemprop_search_space(spec.split("+"))
    if isinstance(spec, dict):
        return spec
    return chemprop_search_space(tuple(spec))


def merge_best_config(hyperparameters: dict, best_config: dict) -> dict:
    """Merge the winning search config into the base hyperparameters (phase-2 config).

    Drops the ``hpo`` block (search is done) and, when ``max_lr`` was searched, ties
    ``init_lr``/``final_lr`` to it (one-tenth of ``max_lr``) so the Noam schedule
    stays well-ordered.

    Args:
        hyperparameters: the user's hyperparameters dict (may include the ``hpo`` block).
        best_config: the winning config from the search.

    Returns:
        dict: hyperparameters to publish the tuned model with — no ``hpo`` block.
    """
    merged = {k: v for k, v in hyperparameters.items() if k != "hpo"}
    merged.update(best_config)
    if "max_lr" in best_config:
        merged["init_lr"] = best_config["max_lr"] / 10.0
        merged["final_lr"] = best_config["max_lr"] / 10.0
    return merged


def run_chemprop_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    target_columns,
    smiles_column: str,
    output_dir: str | None = None,
) -> dict:
    """Phase-1 chemprop hyperparameter search; returns the phase-2 hyperparameters.

    The caller passes the already-split training frame and the held-out ``validation``
    frame (e.g. the template's ``split_validation_set`` output). Each trial trains a full
    ``n_folds`` ensemble — the same regime the winner is published in — scored on the
    held-out set (``holdout_mae``) or out-of-fold (``cv_mae``) when ``val_df`` is empty.
    The winner is merged into ``base_hyperparameters``. Writes ``best_config.json`` +
    ``hpo_trials.csv`` to ``output_dir`` when given.

    Cost is kept near single-fold for weak configs by reporting the running objective
    after every fold: the harness prunes a trial that is already off the pace once
    ``FOLD_PRUNE_WARMUP`` folds are in, so only promising configs pay for the full
    ensemble.

    v1 scope: regression, SMILES-only features; the objective is the primary target's
    MAE. Extra descriptors / bounded loss / multi-task featurization are not exercised
    during search — the phase-2 publish still uses the full template.
    """
    import json

    import pandas as pd

    from workbench.endpoints.chemprop_utils import create_molecule_datapoints
    from workbench.endpoints.inference import get_split_indices
    from workbench.training.hpo_harness import run_search

    def _rdkit_valid(df):
        """Drop rows RDKit can't parse, so predictions stay aligned with the target array."""
        df = df.dropna(subset=[smiles_column]).reset_index(drop=True)
        _, valid = create_molecule_datapoints(df[smiles_column].tolist())
        return df.iloc[valid].reset_index(drop=True)

    # The caller's holdout frame is split off *before* the template's own valid-SMILES
    # filter, so it is filtered here rather than assumed clean.
    train_df, val_df = _rdkit_valid(train_df), _rdkit_valid(val_df)

    # Same folds the template would build for this config, so a trial's ensemble matches
    # the published one. Scaffold is the SMILES-feature default (literature-favored;
    # random splits leak near-duplicate scaffolds across train/val).
    n_folds = int(hpo_block.get("n_folds", base_hyperparameters.get("n_folds", 5)))
    strategy = base_hyperparameters.get("split_strategy", "scaffold")
    folds = get_split_indices(
        train_df,
        n_splits=n_folds,
        strategy=strategy,
        smiles_column=smiles_column,
        test_size=0.2,
        random_state=base_hyperparameters.get("seed", 42),
        butina_cutoff=base_hyperparameters.get("butina_cutoff", 0.4),
    )
    if len(val_df):
        metric, where = "holdout_mae", f"held-out validation set ({len(val_df)} rows)"
    else:
        metric, where = "cv_mae", f"out-of-fold {strategy} splits (no validation_ids)"
    print(
        f"[hpo] objective = {metric} on {where}; {n_folds}-fold ensemble per trial, " f"{len(train_df)} training rows"
    )

    space = resolve_search_space(hpo_block.get("search_space"))
    backend = hpo_block.get("backend", "auto")
    max_parallel = max(1, hpo_block.get("max_parallel", 1))
    # Every dataloader worker is a process, so concurrent trials have to share the vCPUs —
    # oversubscribing them starves the GPU instead of feeding it.
    num_workers = max(1, min((os.cpu_count() or 4) // max_parallel, 8))
    print(f"[hpo] {max_parallel} concurrent trial(s), {num_workers} dataloader workers each")

    trial_fn = _make_trial_fn(
        train_df, folds, val_df, base_hyperparameters, target_columns, smiles_column, metric, num_workers
    )
    result = run_search(
        trial_fn,
        space,
        n_trials=hpo_block.get("n_trials", 40),
        backend=backend,
        max_parallel=max_parallel,
        metric=metric,
        mode="min",
        prune_warmup=FOLD_PRUNE_WARMUP,
        seed=base_hyperparameters.get("seed", 42),
        # A trial uses ~5% of an L4's memory and ~46% of its compute, so packing two per
        # GPU roughly saturates one without spilling. Ray only.
        resources_per_trial={"gpu": hpo_block.get("gpus_per_trial", 0.5)} if backend != "optuna" else None,
    )
    print(f"[hpo] best {metric}={result.best_value:.4f}  best_config={result.best_config}")

    if output_dir:
        with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
            json.dump(
                {"metric": metric, "best_value": result.best_value, "best_config": result.best_config}, fp, indent=2
            )
        pd.DataFrame(result.trials).to_csv(os.path.join(output_dir, "hpo_trials.csv"), index=False)

    return merge_best_config(base_hyperparameters, result.best_config)


def _make_trial_fn(train_df, folds, val_df, base_hyperparameters, target_columns, smiles_column, metric, num_workers):
    """Build the ensemble chemprop ``trial_fn`` (closes over the folds and eval data).

    Each trial trains one model per fold through
    :func:`workbench.training.chemprop_core.train_chemprop_fold` — the same function the
    template publishes with, so a trial's members are built, early-stopped, and
    best-checkpoint-selected identically — and scores the ensemble the way it is deployed:

    * ``holdout_mae`` — every member predicts ``val_df``; the objective is the MAE of the
      members' mean prediction, i.e. the real ensemble's held-out error.
    * ``cv_mae`` — each member is scored on its own out-of-fold rows and the objective is
      the mean across folds.

    The running objective is reported after each fold so the harness can prune a config
    that is already off the pace, which keeps weak trials near single-fold cost.

    Epoch-level early stopping stays with chemprop's ``EarlyStopping`` callback; the
    harness prunes at fold granularity.
    """
    import tempfile

    import numpy as np

    from workbench.training.chemprop_core import FoldSpec, predict_chemprop_frame, train_chemprop_fold

    target_columns = list(target_columns)
    holdout = bool(len(val_df))
    all_y = train_df[target_columns].to_numpy(dtype=float)
    eval_y = val_df[target_columns].to_numpy(dtype=float)[:, 0] if holdout else None

    def trial_fn(config, report):
        spec = FoldSpec(
            hyperparameters=merge_best_config(base_hyperparameters, config),
            smiles_column=smiles_column,
            n_targets=len(target_columns),
            model_type="uq_regressor",
            enable_progress_bar=False,
            verbose=False,
            num_workers=num_workers,
        )

        member_preds, fold_maes = [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)
            # Checkpoints are scratch: the trial keeps the model object, not the file.
            with tempfile.TemporaryDirectory() as ckpt_dir:
                mpnn, fold_preds = train_chemprop_fold(
                    spec,
                    train_df.iloc[tr_idx].reset_index(drop=True),
                    fold_val_df,
                    all_y[tr_idx],
                    all_y[va_idx],
                    fold_idx=fold_idx,
                    checkpoint_dir=ckpt_dir,
                )

            if holdout:
                # Each member predicts the held-out rows; the objective is the ensemble's
                # mean prediction. The fold's own val rows drive early stopping only, so
                # the objective set never influences model selection.
                member_preds.append(predict_chemprop_frame(mpnn, spec, val_df)[:, 0])
                running = float(np.mean(np.abs(np.mean(member_preds, axis=0) - eval_y)))
            else:
                fold_maes.append(float(np.mean(np.abs(fold_preds[:, 0] - all_y[va_idx][:, 0]))))
                running = float(np.mean(fold_maes))

            # Fold-granular pruning: a config already off the pace stops here rather than
            # paying for the remaining members.
            report(step=fold_idx + 1, **{metric: running})

        return running

    return trial_fn
