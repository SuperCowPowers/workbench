"""Chemprop hyperparameter-search objective + default search space.

The chemprop adapter for :mod:`workbench.training.hpo_runner`: a default search space
(this module) and, per trial, a full ``n_folds`` ensemble scored as pooled out-of-fold MAE
(the search *objective*). Scoring a trial in the regime the winner is published in is the
point — a config selected as a lone model does not carry over to an ensemble. The runner
owns everything around that: fold construction, the baseline trial, the artifacts.

The caller's own hyperparameters run as one of the trials, so a search that finds nothing
publishes them unchanged — which matters because the literature finds chemprop HPO is
roughly a coin-flip against defaults on small sets (Tetko et al., J Cheminform 2024) and no
help out-of-distribution (BOOM, arXiv:2505.01912).

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
from workbench.training.hpo_runner import HpoAdapter, pooled_mae, run_hpo

# Default per-knob search space, grouped like chemprop's own hpopt keywords. Both groups
# are searched by default. Everything else — uq_version, max_epochs, patience,
# split_strategy, criterion, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    "basic": {
        # depth follows chemprop's canonical search space; hidden_dim spans 100-2400, and
        # small datasets select capacity at the low end.
        #
        # The head is one knob. ffn_hidden_dim is a dash-string per-layer shape (the pytorch
        # `layers` convention), so its length sets the depth and its values set the widths —
        # a spelling that covers chemprop's (width, ffn_num_layers) pair and adds the tapered
        # heads the pair cannot express. ffn_num_layers is therefore never searched: it is
        # ignored for a shape, so a searched value would be recorded but never built. Every
        # option is a scalar cell in the search records, so the knob plots directly.
        #
        # Ten options, split evenly between uniform widths and tapered shapes, and spread
        # across depth 1-3 so width and depth are not confounded. Ten levels is about the
        # ceiling at a 60-trial budget — a categorical wants several draws per level.
        # The default is the template's untuned head.
        "depth": IntRange(2, 6, 1, default=5),
        "hidden_dim": IntRange(100, 2400, 100, default=700),
        "ffn_hidden_dim": Choice(
            [
                "300",
                "600",
                "1200",
                "1800",
                "300-300",
                "300-100",
                "512-128",
                "1024-256",
                "512-128-32",
                "1024-256-64",
            ],
            default="300-300",
        ),
    },
    # The optimizer knobs the chemprop maintainers recommend tuning. batch_size and LR
    # interact (they scale together), so they search as one group. The default matches
    # chemprop's own (64) and the template's, so an untuned model and the search's baseline
    # train alike; on ADMET-scale sets the optimizers in the literature pick the bottom of
    # whatever batch range they are offered.
    #
    # The ceiling is 512. Activation memory goes as depth x hidden_dim x bonds-per-batch, and
    # 1024 pushed the product past a shared card: measured over one 60-trial search, every
    # failure sat at 1024 and none below it, while 13 trials at hidden_dim>=1600 (up to 2300)
    # ran clean at smaller batches — capacity was never the problem, the product was. It also
    # costs nothing to drop: B_opt scales with dataset size (Smith & Le, arXiv:1710.06451), so
    # 1024 is well past useful for a few thousand molecules.
    #
    # init_lr/final_lr are tied to max_lr in merge_best_config; searched independently they
    # can produce init > max, which the Noam schedule rejects.
    "optimizer": {
        "max_lr": FloatRange(1e-4, 5e-3, log=True, default=1e-3),
        "batch_size": Choice([64, 128, 256, 512], default=64),
    },
}

# Knobs outside the default space — the working list for when this is revisited, ordered by
# expected payoff. The full cross-file backlog (infra + correctness items too) lives in the
# "Backlog / follow-ups" section of docs/planning/hpo_support.md; this comment covers only
# the search-space knobs. The optimizer group carries the search: `max_lr` holds 0.46-0.64 of
# the knob importance in every run so far, `hidden_dim` and `batch_size` behind it, the FFN
# head second-order. Capacity search alone is a modest lever — Yang et al. 2019 measured
# ~2-5% from HPO on most datasets. Ensembling is the big one, and the n_folds publish already
# captures it.
#
#   * `dropout` — chemprop searches it ({0.0,0.05,…,0.4}) and ensemble-scored trials can now
#     select it honestly (a regularization knob is only meaningful against the ensemble it
#     ships in). Low priority — coupled to the ensemble, small effect.
#   * featurization (atom-featurizer mode ORGANIC; RIGR for small datasets) — not in
#     chemprop's "all" keyword, and a template featurizer swap rather than a search range, so
#     it lands as its own feature. RIGR's invariances don't suit ADMET endpoints.
#   * `aggregation` (mean/sum/norm) — searchable in chemprop but unreachable from here:
#     `chemprop_core.build_mpnn_model` constructs `NormAggregation` directly, so tuning it
#     requires a model-construction change first.
#   * `activation`, `aggregation_norm` — the remainder of chemprop's "all" keyword; smallest
#     expected effect.
#
# Each added knob costs trials: the default space is already 6-dimensional and every trial
# trains a full ensemble.


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


def validate_search_space(space: dict) -> dict:
    """Check a resolved space for knobs that cannot both be honored, returning it unchanged.

    ``ffn_num_layers`` only reaches the model when ``ffn_hidden_dim`` is a plain int width;
    a shape carries its own depth. Searching the two together means every shape trial
    records a layer count it never built, which lands in the trial plots, the knob
    importance, and the published winner's hyperparameters. Either search widths as ints
    beside ``ffn_num_layers``, or search shapes alone.

    Args:
        space: a resolved ``{knob: Spec}`` space.

    Returns:
        dict: the same space.

    Raises:
        ValueError: if ``ffn_num_layers`` is searched beside shape-valued widths.
    """
    width = space.get("ffn_hidden_dim")
    shapes = [option for option in getattr(width, "options", ()) if not isinstance(option, int)]
    if "ffn_num_layers" in space and shapes:
        raise ValueError(
            f"ffn_num_layers cannot be searched beside shaped ffn_hidden_dim options ({shapes}): a shape "
            "sets its own layer count, so the sampled ffn_num_layers would be recorded but never built. "
            "Drop ffn_num_layers, or give ffn_hidden_dim int widths only."
        )
    return space


def resolve_search_space(spec) -> dict:
    """Resolve an ``hpo['search_space']`` value into a ``{knob: Spec}`` space.

    Accepts a shorthand string (``"basic"``, ``"basic+optimizer"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to all groups (``basic+optimizer``). Custom spaces are checked by
    :func:`validate_search_space`.
    """
    if spec is None:
        return chemprop_search_space()
    if isinstance(spec, str):
        return chemprop_search_space(spec.split("+"))
    if isinstance(spec, dict):
        # A dict of dicts is the JSON wire form; a dict of Specs is already a space
        if spec and all(isinstance(v, dict) for v in spec.values()):
            return validate_search_space(SearchSpace.from_dict(spec))
        return validate_search_space(spec)
    return chemprop_search_space(tuple(spec))


def merge_best_config(hyperparameters: dict, best_config: dict) -> dict:
    """Merge the winning search config into the base hyperparameters (phase-2 config).

    Drops the ``hpo`` block (search is done) and, when ``max_lr`` was *changed*, ties
    ``init_lr``/``final_lr`` to it (one-tenth of ``max_lr``) so the Noam schedule stays
    well-ordered. A config carrying the caller's own ``max_lr`` — the seeded baseline does,
    since it names every searched knob — leaves their schedule alone.

    Args:
        hyperparameters: the user's hyperparameters dict (may include the ``hpo`` block).
        best_config: the winning config from the search.

    Returns:
        dict: hyperparameters to publish the tuned model with — no ``hpo`` block.
    """
    merged = {k: v for k, v in hyperparameters.items() if k != "hpo"}
    merged.update(best_config)
    if "max_lr" in best_config and best_config["max_lr"] != hyperparameters.get("max_lr"):
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

    def split_kwargs(self) -> dict:
        return {"smiles_column": self.smiles_column}

    def baseline_config(self, config, hyperparameters) -> dict:
        """Spell the caller's FFN head as the shape the space samples.

        The template takes an int width beside ``ffn_num_layers``; the space takes a
        per-layer shape, and only shapes are samplable. Same architecture either way — but
        an int is not one of the Choice's options, so an untranslated baseline cannot be
        seeded at all.
        """
        width = config.get("ffn_hidden_dim")
        if not isinstance(width, int):
            return config
        n_layers = int(hyperparameters.get("ffn_num_layers", 1))
        return {**config, "ffn_hidden_dim": "-".join([str(width)] * n_layers)}

    def merge_config(self, hyperparameters, config) -> dict:
        return merge_best_config(hyperparameters, config)

    # A trial is a full D-MPNN ensemble, but the fold ladder stops most of them at the
    # first or second member, so the budget buys far more configs than the fold count suggests.
    default_n_trials = 60

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

    def make_trial_fn(self, *, train_df, folds, hyperparameters, concurrency):
        """Build the ensemble chemprop ``trial_fn`` (closes over the folds).

        Each trial trains one model per fold through
        :func:`workbench.training.chemprop_core.train_chemprop_fold` — the same function the
        template publishes with, so a trial's members are built, early-stopped, and
        best-checkpoint-selected identically. Each member predicts its own out-of-fold rows
        and the objective is one MAE over those pooled predictions.

        The pool is reported after every fold, so the harness can stop a trial that is
        already off the pace. Fold order is fixed, so every trial at a given fold was scored
        on the same molecules; and pooling over the folds run so far estimates the same
        quantity the last fold reports, on a subsample of it.

        Epoch-level early stopping stays with chemprop's ``EarlyStopping`` callback.
        """
        import tempfile

        from workbench.training.chemprop_core import FoldSpec, train_chemprop_fold

        num_workers = _dataloader_workers(concurrency)
        print(f"[hpo] {num_workers} dataloader workers per trial at {concurrency} concurrent")

        target_columns = self.target_columns
        all_y = train_df[target_columns].to_numpy(dtype=float)
        # Trials must train the way the published model does, weights included.
        sample_weight = train_df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train_df.columns else None

        def trial_fn(config, report):
            spec = FoldSpec(
                hyperparameters=merge_best_config(hyperparameters, config),
                smiles_column=self.smiles_column,
                n_targets=len(target_columns),
                target_columns=target_columns,
                enable_progress_bar=False,
                verbose=False,
                num_workers=num_workers,
                concurrent=concurrency > 1,
                **self.model_shape,
            )

            oof_pred, oof_true = [], []
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                fold_val_df = train_df.iloc[va_idx].reset_index(drop=True)
                # Checkpoints are scratch: the trial keeps the model object, not the file.
                with tempfile.TemporaryDirectory() as ckpt_dir:
                    _, fold_preds = train_chemprop_fold(
                        spec,
                        train_df.iloc[tr_idx].reset_index(drop=True),
                        fold_val_df,
                        all_y[tr_idx],
                        all_y[va_idx],
                        fold_idx=fold_idx,
                        checkpoint_dir=ckpt_dir,
                        train_sample_weight=None if sample_weight is None else sample_weight[tr_idx],
                    )

                oof_pred.append(fold_preds[:, 0])
                oof_true.append(all_y[va_idx][:, 0])
                running = pooled_mae(oof_pred, oof_true)
                report(fold_idx + 1, running)

            return running

        return trial_fn


def run_chemprop_hpo(
    train_df,
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

    See :func:`workbench.training.hpo_runner.run_hpo` for the search contract and
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
