"""Chemprop hyperparameter-search objective + default search space.

Drives the framework-agnostic :mod:`workbench.training.hpo_harness` for chemprop: a
default search space (this module) and, per trial, a full ``n_folds`` ensemble scored as
``holdout_mae`` / ``cv_mae`` (the search *objective*). Scoring a trial in the regime the
winner is published in is the point — a config selected as a lone model does not carry
over to an ensemble. Training-only; imported **only inside the chemprop template's
``__main__``** (deferred).

Selection is two-stage: the search shortlists, then :func:`_rerank_finalists` re-scores the
finalists *and the caller's baseline* on fresh trainings and picks from those. The search's
own winning value is the minimum of many noisy estimates and so is optimistically biased;
carrying the baseline through the re-rank is what bounds the feature's downside, which
matters because the literature finds chemprop HPO is roughly a coin-flip against defaults
on small sets (Tetko et al., J Cheminform 2024) and no help out-of-distribution (BOOM,
arXiv:2505.01912).

Parity: each trial builds its model through
:func:`workbench.training.chemprop_core.build_mpnn_model` — the same builder the
template uses to publish the winner — so a searched config maps to the identical
architecture and optimizer schedule.

Only the pure pieces here (the search space and config merge) import without
chemprop; the search entry point and per-trial train defer their chemprop/chemprop_core
imports so this module stays importable for unit tests.
"""

from __future__ import annotations

import json
import os

from workbench.training.hpo_harness import Choice, FloatRange, IntRange

# Trials report once per completed fold, so a trial is eligible for pruning only after its
# second member has trained. One fold is too noisy a basis to kill a config on: a config can
# lag on a single scaffold fold and still make the better ensemble.
FOLD_PRUNE_WARMUP = 2

# Finalists re-scored in phase 1.5 (plus the baseline, always). See _rerank_finalists.
RERANK_TOP_K = 5

# Added to the seed for the re-rank pass. Trials are deterministic (train_chemprop_fold seeds
# on `seed + fold_idx`), so re-scoring a config at the search seed would replay the search's
# number rather than draw an independent one.
RERANK_SEED_OFFSET = 1000

# Default per-knob search space, grouped like chemprop's own hpopt keywords. Both groups
# are searched by default. Everything else — uq_version, max_epochs, patience,
# split_strategy, criterion, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    "basic": {
        # depth / ffn_num_layers ranges are chemprop's canonical search space. hidden_dim
        # extends chemprop's 300 floor down to 100 (the AqSol optimum sat on that floor).
        # ffn_hidden_dim additionally offers tapered heads (lists), which chemprop's own
        # space can't express — ffn_num_layers is ignored when a tapered list is chosen
        # (the list length sets the depth).
        "depth": IntRange(2, 6, 1),
        "hidden_dim": IntRange(100, 2400, 100),
        "ffn_num_layers": IntRange(1, 3, 1),
        "ffn_hidden_dim": Choice([300, 600, 1200, 1800, 2400, [1024, 256, 64], [512, 128]]),
    },
    # The optimizer knobs the chemprop maintainers recommend tuning. batch_size and LR
    # interact (they scale together), so they search as one group. The batch_size ceiling is
    # set by optimization dynamics, not GPU memory — a 60-trial 8-concurrent search peaks at
    # ~22% of a 4x L4 box. init_lr/final_lr are tied to max_lr in merge_best_config rather
    # than searched independently (independent search can produce init > max, which the Noam
    # schedule rejects).
    "lr": {
        "max_lr": FloatRange(1e-4, 5e-3, log=True),
        "warmup_epochs": IntRange(2, 10, 2),
        "batch_size": Choice([32, 64, 128, 256, 512]),
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


def chemprop_search_space(groups=("basic", "lr")) -> dict:
    """Build the default chemprop search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture capacity) and/or
            ``"lr"`` (the learning-rate schedule + batch size). Both by default.

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
    Defaults to all groups (``basic+lr``).
    """
    if spec is None:
        return chemprop_search_space()
    if isinstance(spec, str):
        return chemprop_search_space(spec.split("+"))
    if isinstance(spec, dict):
        return spec
    return chemprop_search_space(tuple(spec))


def _use_holdout(requested_metric, n_val_rows: int) -> bool:
    """Whether the designated validation rows should drive the search objective.

    Defaults to out-of-fold scoring, leaving any validation rows out of the objective. They
    stay out of training either way (the template split them off) — this only decides
    whether they *drive the search*. Out-of-fold is the safe default because
    ``validation_ids`` usually marks a benchmark set, and tuning on one makes its own final
    score optimistic and unfair against models that never saw those labels. Opt in with
    ``hpo["metric"]="holdout_mae"`` when the holdout exists to be tuned toward.

    Args:
        requested_metric: the ``hpo["metric"]`` value, or None for the default.
        n_val_rows: how many validation rows the caller designated.

    Returns:
        bool: True to score ``holdout_mae`` on the validation rows, False for ``cv_mae``.
    """
    if requested_metric not in (None, "holdout_mae", "cv_mae"):
        raise ValueError(f"hpo['metric'] must be 'holdout_mae' or 'cv_mae', got {requested_metric!r}")
    if requested_metric == "holdout_mae":
        if not n_val_rows:
            raise ValueError("hpo['metric']='holdout_mae' needs validation_ids, but no validation rows were designated")
        return True
    return False


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
    Writes ``best_config.json``, ``hpo_trials.csv`` and ``hpo_rerank.csv`` to ``output_dir``
    when given.

    Cost is kept near single-fold for weak configs by reporting the running objective
    after every fold: the harness prunes a trial that is already off the pace once
    ``FOLD_PRUNE_WARMUP`` folds are in, so only promising configs pay for the full
    ensemble.

    The search does not pick the winner on its own — its finalists go through
    :func:`_rerank_finalists`, which re-scores them and the *baseline* on fresh trainings
    and selects on those. The published config is whichever wins there, so a search that
    found nothing real publishes the caller's own hyperparameters unchanged. Disable with
    ``hpo["rerank_top_k"] = 0``.

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

    # The objective is the PRIMARY target's MAE, but the template keeps a row when any target
    # is non-NaN — so a multi-target frame can arrive with unlabeled primary targets. Scoring
    # nanmean-skips them; this check catches the degenerate case up front rather than after a
    # full search (a NaN objective fails every trial, which surfaces as an opaque crash).
    primary = list(target_columns)[0]
    labeled = int(train_df[primary].notna().sum())
    if labeled == 0:
        raise ValueError(f"HPO objective needs non-NaN values in the primary target {primary!r}; found none.")
    if labeled < len(train_df):
        print(
            f"[hpo] {len(train_df) - labeled} of {len(train_df)} training rows have no {primary}; "
            "trained on, but excluded from scoring"
        )

    n_val_rows = len(val_df)
    if not _use_holdout(hpo_block.get("metric"), n_val_rows):
        # Emptying it here is what routes the trial/re-rank scoring to out-of-fold.
        val_df = val_df.iloc[:0]

    # Same folds the template would build for this config, so a trial's ensemble matches
    # the published one. Scaffold is the SMILES-feature default (literature-favored;
    # random splits leak near-duplicate scaffolds across train/val).
    publish_folds = int(base_hyperparameters.get("n_folds", 5))
    n_folds = int(hpo_block.get("n_folds", publish_folds))
    if n_folds != publish_folds:
        print(
            f"[hpo] WARNING: searching/re-ranking with {n_folds}-fold ensembles but publishing with "
            f"{publish_folds} — the winner is selected in a different regime than it ships in. "
            "hpo['n_folds'] is for cheap validation runs, not real searches."
        )
    strategy = base_hyperparameters.get("split_strategy", "scaffold")
    seed = base_hyperparameters.get("seed", 42)
    split_kwargs = dict(
        n_splits=n_folds,
        strategy=strategy,
        smiles_column=smiles_column,
        test_size=0.2,
        butina_cutoff=base_hyperparameters.get("butina_cutoff", 0.4),
    )
    folds = get_split_indices(train_df, random_state=seed, **split_kwargs)
    if len(val_df):
        metric, where = "holdout_mae", f"held-out validation set ({len(val_df)} rows)"
    else:
        # Say which it is: no rows were designated, or rows were designated and deliberately
        # kept out of the objective. The two mean very different things when reading a log.
        excluded = (
            f"{n_val_rows} validation rows held out of training AND out of the search"
            if n_val_rows
            else "no validation_ids"
        )
        metric, where = "cv_mae", f"out-of-fold {strategy} splits ({excluded})"
    print(
        f"[hpo] objective = {metric} on {where}; {n_folds}-fold ensemble per trial, " f"{len(train_df)} training rows"
    )

    space = resolve_search_space(hpo_block.get("search_space"))
    backend = hpo_block.get("backend", "auto")
    max_parallel = max(1, hpo_block.get("max_parallel", 1))
    num_workers = _dataloader_workers(max_parallel)
    print(f"[hpo] {max_parallel} concurrent trial(s), {num_workers} dataloader workers each")

    # A trial uses ~5% of an L4's memory and ~46% of its compute, so packing two per GPU
    # roughly saturates one without spilling. Ray only.
    resources = {"gpu": hpo_block.get("gpus_per_trial", 0.5)} if backend != "optuna" else None

    trial_fn = _make_trial_fn(
        train_df, folds, val_df, base_hyperparameters, target_columns, smiles_column, metric, num_workers
    )
    result = run_search(
        trial_fn,
        space,
        n_trials=hpo_block.get("n_trials", 60),
        backend=backend,
        max_parallel=max_parallel,
        metric=metric,
        mode="min",
        prune_warmup=FOLD_PRUNE_WARMUP,
        seed=seed,
        resources_per_trial=resources,
    )
    print(f"[hpo] search best {metric}={result.best_value:.4f}  config={result.best_config}")

    # Phase 1.5 refines a result the search has already produced, so its failure degrades to
    # the unrefined winner rather than discarding a search that has already run to completion.
    try:
        best_config, rerank = _rerank_finalists(
            result,
            top_k=int(hpo_block.get("rerank_top_k", RERANK_TOP_K)),
            train_df=train_df,
            val_df=val_df,
            folds=folds,
            split_kwargs=split_kwargs,
            base_hyperparameters=base_hyperparameters,
            target_columns=target_columns,
            smiles_column=smiles_column,
            metric=metric,
            num_workers=num_workers,
            seed=seed,
            backend=backend,
            max_parallel=max_parallel,
            resources=resources,
        )
    except Exception as exc:
        print(f"[hpo] re-rank FAILED ({exc!r}); publishing the search winner unrefined")
        best_config, rerank = result.best_config, {}

    if output_dir:
        with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
            json.dump(
                {
                    "metric": metric,
                    "best_config": best_config,
                    # best_value and baseline_value share the re-rank's basis, so their
                    # difference is the real margin the publish decision turned on.
                    "best_value": rerank.get("best_value"),
                    "baseline_value": rerank.get("baseline_value"),
                    # Phase 1's own number. When rerank_fresh_split is true the re-rank scored
                    # on a different fold partition, so this is NOT comparable to the two
                    # above — partitions differ in difficulty.
                    "search_best_value": result.best_value,
                    "search_best_config": result.best_config,
                    "rerank_fresh_split": rerank.get("fresh_split", False),
                    "rerank": rerank.get("candidates", []),
                },
                fp,
                indent=2,
                default=str,
            )
        pd.DataFrame(result.trials).to_csv(os.path.join(output_dir, "hpo_trials.csv"), index=False)
        if rerank.get("candidates"):
            pd.DataFrame(rerank["candidates"]).to_csv(os.path.join(output_dir, "hpo_rerank.csv"), index=False)

    return merge_best_config(base_hyperparameters, best_config)


def _dataloader_workers(concurrency: int) -> int:
    """Dataloader workers per trial at a given concurrency.

    Every worker is a process, so concurrent trials share the vCPUs — oversubscribing them
    starves the GPU instead of feeding it. Phase 1.5 runs fewer trials at once than the
    search, so it recomputes this rather than inheriting the search's tighter budget.
    """
    return max(1, min((os.cpu_count() or 4) // max(1, concurrency), 8))


def _trial_completed(trial: dict) -> bool:
    """Whether a trial ran to completion (not pruned) — both backends' record shapes."""
    if "completed" in trial:  # ray
        return bool(trial["completed"])
    return trial.get("state") == "COMPLETE"  # optuna


def _shortlist_configs(trials, top_k: int) -> list:
    """The ``top_k`` distinct configs among completed trials, best (lowest) first."""
    ranked, seen = [], set()
    for trial in trials:
        value = trial.get("value")
        if value is None or not _trial_completed(trial):
            continue
        config = trial.get("config") or {}
        key = json.dumps(config, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((value, config))
    ranked.sort(key=lambda pair: pair[0])
    return [config for _, config in ranked[:top_k]]


def _rerank_finalists(
    result,
    *,
    top_k,
    train_df,
    val_df,
    folds,
    split_kwargs,
    base_hyperparameters,
    target_columns,
    smiles_column,
    metric,
    num_workers,
    seed,
    backend,
    max_parallel,
    resources,
):
    """Phase 1.5 — re-score the search's finalists, plus the baseline, on fresh trainings.

    The search reports the *minimum* over many noisy estimates, so its winning value is
    optimistically biased and the winner may simply have drawn the luckiest evaluation
    (winner's curse). Re-scoring a shortlist independently and selecting on *that* is the
    fix. The user's own hyperparameters ride along as a candidate, which is what keeps HPO
    from making the model worse: a search that found nothing real loses to the baseline and
    the baseline is what gets published. Ties go to the baseline.

    Independence comes from a fresh seed (trials are deterministic, so the search seed would
    replay rather than redraw) and, in ``cv_mae`` mode, a fresh fold partition as well — the
    split is part of what the search selected against there. In ``holdout_mae`` mode the
    holdout is the user's fixed OOD set and is deliberately reused, so that pass removes the
    training-stochasticity component of the bias but not the holdout-sampling component.

    Returns:
        tuple: ``(best_config, info)`` — the config to publish, and a dict carrying
        ``candidates`` (the per-candidate record), the winning and baseline values on the
        re-rank's own basis, and ``fresh_split``. ``info`` is empty when the re-rank is
        disabled or nothing scored.
    """
    from workbench.endpoints.inference import get_split_indices
    from workbench.training.hpo_harness import evaluate_configs

    if top_k <= 0:
        print("[hpo] re-rank disabled (rerank_top_k=0); publishing the search winner")
        return result.best_config, {}

    # The baseline is the empty config: merged over base_hyperparameters it changes nothing.
    candidates = [{}] + _shortlist_configs(result.trials, top_k)
    if len(candidates) == 1:
        print("[hpo] re-rank: no completed trials to re-rank; publishing the search winner")
        return result.best_config, {}

    rerank_seed = seed + RERANK_SEED_OFFSET
    rerank_folds = folds
    if metric == "cv_mae":
        rerank_folds = get_split_indices(train_df, random_state=rerank_seed, **split_kwargs)
    print(
        f"[hpo] re-rank: {len(candidates)} candidates (baseline + {len(candidates) - 1} finalists) "
        f"on fresh {metric}" + (" and a fresh fold partition" if metric == "cv_mae" else "")
    )

    # At most one slot per candidate runs at a time, so the vCPUs divide fewer ways here than
    # during the search — recompute rather than inherit the search's tighter worker budget.
    rerank_parallel = min(max_parallel, len(candidates))
    rerank_fn = _make_trial_fn(
        train_df,
        rerank_folds,
        val_df,
        base_hyperparameters,
        target_columns,
        smiles_column,
        metric,
        _dataloader_workers(rerank_parallel),
    )

    def evaluate(config, index):
        return rerank_fn({**config, "seed": rerank_seed}, lambda **_: None)

    values = evaluate_configs(
        evaluate, candidates, backend=backend, max_parallel=rerank_parallel, resources_per_trial=resources
    )
    if values[0] is None:
        # Without a baseline score the comparison is finalist-vs-finalist, so the guarantee
        # that HPO can't ship something worse than the caller's own hyperparameters is gone.
        print("[hpo] re-rank WARNING: the baseline failed to score — publishing WITHOUT the baseline guard")
    # Candidate i>0 is the search's rank-i config (the shortlist is best-first), so the label
    # names that rank rather than a position in this list.
    rows = [
        {"candidate": "baseline" if i == 0 else f"search_rank_{i}", "config": c, metric: v}
        for i, (c, v) in enumerate(zip(candidates, values))
    ]
    info = {"candidates": rows, "fresh_split": metric == "cv_mae", "baseline_value": values[0], "best_value": None}

    scored = [(v, i) for i, v in enumerate(values) if v is not None]
    if not scored:
        print("[hpo] re-rank: no candidate scored; publishing the search winner")
        return result.best_config, info

    # min() over (value, index) returns the first minimum, and the baseline is index 0 — so
    # an exact tie is resolved in the baseline's favor.
    best_value, best_index = min(scored)
    baseline_value = values[0]
    info["best_value"] = best_value
    if best_index == 0:
        print(f"[hpo] re-rank winner: BASELINE at {metric}={best_value:.4f} — no searched config beat it")
    else:
        margin = f", {100 * (baseline_value - best_value) / baseline_value:+.1f}% vs baseline" if baseline_value else ""
        print(f"[hpo] re-rank winner: search_rank_{best_index} at {metric}={best_value:.4f}{margin}")
        print(f"[hpo] published config: {candidates[best_index]}")
    return candidates[best_index], info


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
