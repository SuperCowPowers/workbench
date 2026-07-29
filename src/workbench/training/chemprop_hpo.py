"""Chemprop hyperparameter-search objective + default search space.

The chemprop adapter for :mod:`workbench.training.hpo_runner`: a default search space
(this module) and, per trial, a full ``n_folds`` ensemble scored as ``holdout_mae`` /
``cv_mae`` (the search *objective*). Scoring a trial in the regime the winner is
published in is the point — a config selected as a lone model does not carry over to an
ensemble. The runner owns everything around that: selection, the re-rank, the artifacts.

Carrying the caller's baseline through the runner's re-rank is what bounds the feature's
downside, which matters because the literature finds chemprop HPO is roughly a coin-flip
against defaults on small sets (Tetko et al., J Cheminform 2024) and no help
out-of-distribution (BOOM, arXiv:2505.01912).

Parity: each trial builds its model through
:func:`workbench.training.chemprop_core.build_mpnn_model` — the same builder the
template uses to publish the winner — so a searched config maps to the identical
architecture and optimizer schedule.

Training-only; imported **only inside the chemprop template's ``__main__``** (deferred).
Only the pure pieces here (the search space and config merge) import without chemprop;
the per-trial train defers its chemprop/chemprop_core imports so this module stays
importable for unit tests.
"""

from __future__ import annotations

import os

from workbench.training.hpo_harness import Choice, FloatRange, IntRange, SearchSpace
from workbench.training.hpo_runner import HpoAdapter, run_hpo

# Default per-knob search space, grouped like chemprop's own hpopt keywords. Both groups
# are searched by default. Everything else — uq_version, max_epochs, patience,
# split_strategy, criterion, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    "basic": {
        # depth / ffn_num_layers follow chemprop's canonical search space. hidden_dim spans
        # 100-2400; small datasets select capacity at the low end. A dash-string
        # ffn_hidden_dim ("1024-256-64", the pytorch `layers` convention) gives a tapered
        # head, which a scalar width cannot express — ffn_num_layers is ignored for a
        # shape (its length sets the depth). Every option is a scalar cell in the search
        # records, so the knob plots directly. Narrow heads dominate on the ADMET sets, so
        # the flat rungs are coarse above 600 and the shapes all taper to a small tail.
        "depth": IntRange(2, 6, 1, default=5),
        "hidden_dim": IntRange(100, 2400, 100, default=700),
        "ffn_num_layers": IntRange(1, 3, 1, default=2),
        "ffn_hidden_dim": Choice([300, 600, 1800, "300-100", "512-128", "512-128-32", "1024-256-64"], default=300),
    },
    # The optimizer knobs the chemprop maintainers recommend tuning. batch_size and LR
    # interact (they scale together), so they search as one group. The batch_size ceiling is
    # bounded by GPU memory under concurrent trials, not by the objective.
    # init_lr/final_lr are tied to max_lr in merge_best_config; searched independently they
    # can produce init > max, which the Noam schedule rejects.
    "optimizer": {
        "max_lr": FloatRange(1e-4, 5e-3, log=True, default=1e-3),
        "batch_size": Choice([64, 128, 256, 512, 1024], default=128),
    },
}

# Knobs outside the default space — the working list for when this is revisited, ordered by
# expected payoff. The full cross-file backlog (infra + correctness items too) lives in the
# "Backlog / follow-ups" section of docs/planning/hpo_support.md; this comment covers only
# the search-space knobs. Capacity search (the `basic` group) is a modest lever: Yang et
# al. 2019 measured ~2-5% from HPO on most datasets, and on AqSol the search drove
# capacity to the floor for a small win. The bigger remaining lever is featurization — and
# ensembling, which the n_folds publish already captures.
#
#   * featurization (atom-featurizer mode ORGANIC; RIGR for small datasets) — the biggest
#     untapped lever and not even in chemprop's "all" keyword. A template featurizer swap,
#     not a search range, so it lands as its own feature rather than a knob here.
#   * `dropout` — chemprop searches it ({0.0,0.05,…,0.4}) and ensemble-scored trials can now
#     select it honestly (a regularization knob is only meaningful against the ensemble it
#     ships in). Low priority — coupled to the ensemble, small effect.
#   * `aggregation` (mean/sum/norm) — searchable in chemprop but unreachable from here:
#     `chemprop_core.build_mpnn_model` constructs `NormAggregation` directly, so tuning it
#     requires a model-construction change first.
#   * `activation`, `aggregation_norm` — the remainder of chemprop's "all" keyword; smallest
#     expected effect.
#
# Each added knob costs trials: the default space is already 7-dimensional, pruning reserves
# the first PRUNE_STARTUP_TRIALS trials as un-pruned baselines, and every trial trains a
# full ensemble.


def chemprop_search_space(groups=("basic", "optimizer")) -> dict:
    """Build the default chemprop search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture capacity) and/or
            ``"optimizer"`` (the learning-rate schedule + batch size). Both by default.

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

    Accepts a shorthand string (``"basic"``, ``"basic+optimizer"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to all groups (``basic+optimizer``).
    """
    if spec is None:
        return chemprop_search_space()
    if isinstance(spec, str):
        return chemprop_search_space(spec.split("+"))
    if isinstance(spec, dict):
        # A dict of dicts is the JSON wire form; a dict of Specs is already a space
        if spec and all(isinstance(v, dict) for v in spec.values()):
            return SearchSpace.from_dict(spec)
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


class ChempropAdapter(HpoAdapter):
    """Trains and scores one chemprop candidate for :func:`run_hpo`.

    ``task`` / ``model_type`` / ``num_classes`` / ``task_weights`` shape the trial models the
    same way the template shapes the published one. ``task_weights`` in particular must be
    forwarded for multi-task runs: without it every trial trains at equal task weight and the
    search optimizes a different loss than the winner ships with.
    """

    def __init__(self, *, target_columns, smiles_column, task, model_type, num_classes, task_weights):
        self.target_columns = list(target_columns)
        self.smiles_column = smiles_column
        self.model_shape = {
            "task": task,
            "model_type": model_type,
            "num_classes": num_classes,
            "task_weights": task_weights,
        }

    def prepare_frame(self, df):
        """Drop rows RDKit can't parse, so predictions stay aligned with the target array."""
        from workbench.endpoints.chemprop_utils import create_molecule_datapoints

        df = df.dropna(subset=[self.smiles_column]).reset_index(drop=True)
        _, valid = create_molecule_datapoints(df[self.smiles_column].tolist())
        return df.iloc[valid].reset_index(drop=True)

    def split_kwargs(self) -> dict:
        return {"smiles_column": self.smiles_column}

    def merge_config(self, hyperparameters, config) -> dict:
        return merge_best_config(hyperparameters, config)

    def resources_per_trial(self, hpo_block, backend):
        """GPU share per trial. Ray only; ``hpo['gpus_per_trial']`` overrides.

        Two trials per card saturates an L4 without spilling for a single-task search. A
        multi-task one gets the whole card: its good configs sit at the top of the capacity
        range and its molecules come from more than one assay, and packing two of those has
        been measured to run a card out of memory — taking the co-scheduled trial with it.
        """
        if backend == "optuna":
            return None
        default = 1.0 if len(self.target_columns) > 1 else 0.5
        return {"gpu": hpo_block.get("gpus_per_trial", default)}

    def make_trial_fn(self, *, train_df, folds, val_df, hyperparameters, metric, concurrency):
        """Build the ensemble chemprop ``trial_fn`` (closes over the folds and eval data).

        Each trial trains one model per fold through
        :func:`workbench.training.chemprop_core.train_chemprop_fold` — the same function the
        template publishes with, so a trial's members are built, early-stopped, and
        best-checkpoint-selected identically — and scores the ensemble the way it is deployed:

        * ``holdout_mae`` — every member predicts ``val_df``; the objective is the MAE of the
          members' mean prediction, i.e. the real ensemble's held-out error.
        * ``cv_mae`` — each member is scored on its own out-of-fold rows and the objective is
          the mean across folds.

        Epoch-level early stopping stays with chemprop's ``EarlyStopping`` callback; the
        runner's harness prunes at fold granularity.
        """
        import tempfile

        import numpy as np

        from workbench.training.chemprop_core import FoldSpec, predict_chemprop_frame, train_chemprop_fold

        num_workers = _dataloader_workers(concurrency)
        print(f"[hpo] {num_workers} dataloader workers per trial at {concurrency} concurrent")

        target_columns = self.target_columns
        holdout = bool(len(val_df))
        all_y = train_df[target_columns].to_numpy(dtype=float)
        eval_y = val_df[target_columns].to_numpy(dtype=float)[:, 0] if holdout else None

        def trial_fn(config, report):
            spec = FoldSpec(
                hyperparameters=merge_best_config(hyperparameters, config),
                smiles_column=self.smiles_column,
                n_targets=len(target_columns),
                enable_progress_bar=False,
                verbose=False,
                num_workers=num_workers,
                **self.model_shape,
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

                # nanmean, not mean: the template keeps a row when ANY target is non-NaN, so a
                # multi-target frame can carry a NaN primary target. Training still uses every
                # row (chemprop masks per-target); only the scoring skips the unlabeled ones.
                if holdout:
                    # Each member predicts the held-out rows; the objective is the ensemble's
                    # mean prediction. The fold's own val rows drive early stopping only, so
                    # the objective set never influences model selection.
                    member_preds.append(predict_chemprop_frame(mpnn, spec, val_df)[:, 0])
                    running = float(np.nanmean(np.abs(np.mean(member_preds, axis=0) - eval_y)))
                else:
                    fold_maes.append(float(np.nanmean(np.abs(fold_preds[:, 0] - all_y[va_idx][:, 0]))))
                    running = float(np.mean(fold_maes))

                # Fold-granular pruning: a config already off the pace stops here rather than
                # paying for the remaining members.
                report(step=fold_idx + 1, **{metric: running})

            return running

        return trial_fn


def run_chemprop_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    target_columns,
    smiles_column: str,
    task: str = "regression",
    model_type: str = "uq_regressor",
    num_classes: int | None = None,
    task_weights=None,
    output_dir: str | None = None,
) -> dict:
    """Run the chemprop hyperparameter search; returns the phase-2 hyperparameters.

    See :func:`workbench.training.hpo_runner.run_hpo` for the search/re-rank contract and
    the ``hpo`` block's keys.

    v1 scope: the objective is the primary target's MAE, computed with ``nanmean`` so
    multi-task rows lacking that target are trained on but not scored. Extra descriptors and
    bounded-loss censoring are NOT exercised during search (they need per-fold data
    plumbing, not just config) — the phase-2 publish uses the full template.
    """
    adapter = ChempropAdapter(
        target_columns=target_columns,
        smiles_column=smiles_column,
        task=task,
        model_type=model_type,
        num_classes=num_classes,
        task_weights=task_weights,
    )
    return run_hpo(
        train_df,
        val_df,
        base_hyperparameters,
        hpo_block,
        adapter=adapter,
        search_space=resolve_search_space(hpo_block.get("search_space")),
        primary_target=list(target_columns)[0],
        output_dir=output_dir,
    )


def _dataloader_workers(concurrency: int) -> int:
    """Dataloader workers per trial at a given concurrency.

    Every worker is a process, so concurrent trials share the vCPUs — oversubscribing them
    starves the GPU instead of feeding it. Phase 1.5 runs fewer trials at once than the
    search, so it recomputes this rather than inheriting the search's tighter budget.
    """
    return max(1, min((os.cpu_count() or 4) // max(1, concurrency), 8))
