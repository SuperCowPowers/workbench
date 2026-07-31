"""Framework-agnostic hyperparameter-search orchestration.

Sits between :mod:`workbench.training.hpo_harness` (the sampler/backend layer) and a
framework module (:mod:`workbench.training.chemprop_hpo`,
:mod:`workbench.training.xgb_hpo`). Everything a search does *around* training a
candidate lives here: objective selection, fold construction, the same-basis baseline,
the phase-1.5 re-rank, and the ``best_config.json`` / ``hpo_trials.csv`` /
``hpo_rerank.csv`` artifacts that :func:`workbench.utils.model_utils.get_hpo_results`
reads.

A framework supplies an :class:`HpoAdapter` — how to train and score one candidate, and
how a winning config merges back into the published hyperparameters. Nothing else about
the framework reaches this module, so the artifact contract is identical across them.

Selection is two-stage: the search shortlists, then :func:`rerank_finalists` re-scores
the finalists *and the caller's baseline* on fresh trainings and picks from those. The
search's own winning value is the minimum of many noisy estimates and so is
optimistically biased; carrying the baseline through the re-rank is what bounds the
feature's downside.

Training-only; imported from inside a model template's ``__main__``.
"""

from __future__ import annotations

import json
import os

# Trials report once per completed fold, so a trial is eligible for pruning only after its
# second member has trained. One fold is too noisy a basis to kill a config on: a config can
# lag on a single scaffold fold and still make the better ensemble.
#
# This also anchors ASHA's rung ladder, which sits at this value times
# PRUNE_REDUCTION_FACTOR ** k — 2 and 4 for a 5-fold search. Raising it moves every rung:
# at 3 the second rung lands at 6, past the last fold, leaving one all-or-nothing cull.
# Check the ladder against n_folds before changing this.
FOLD_PRUNE_WARMUP = 2

# Finalists re-scored in phase 1.5 (plus the baseline, always). See rerank_finalists.
RERANK_TOP_K = 5

# Added to the seed for the re-rank pass. Trials are deterministic, so re-scoring a config at
# the search seed would replay the search's number rather than draw an independent one.
RERANK_SEED_OFFSET = 1000


class HpoAdapter:
    """Framework hooks for :func:`run_hpo`.

    A subclass carries whatever the framework needs to train a candidate (target columns,
    feature list, model shape) and implements :meth:`make_trial_fn`. The rest have working
    defaults.
    """

    def prepare_frame(self, df):
        """Align a frame to what training expects — row filtering, fitted transforms.

        Runs on the training and holdout frames before anything else, so predictions stay
        positionally aligned with the target array. The holdout arrives raw (it is split
        off before the template's preprocessing), while the training frame has already
        been through it — so an override must be idempotent.
        """
        return df

    def split_kwargs(self) -> dict:
        """Extra ``get_split_indices`` kwargs, e.g. the SMILES column for scaffold folds."""
        return {}

    def make_trial_fn(self, *, train_df, folds, val_df, hyperparameters, metric, concurrency):
        """Build the ``(config, report) -> float`` objective for one search pass.

        Each call closes over a specific fold partition, so the search and the re-rank get
        their own. A trial trains the full ensemble the winner is published as and returns
        ``metric`` — ``holdout_mae`` (ensemble mean prediction on ``val_df``) when ``val_df``
        is non-empty, else ``cv_mae`` (mean of the per-fold out-of-fold errors).

        Report the running objective after each fold via ``report(step=..., **{metric: ...})``
        so the harness can prune a config that is already off the pace.

        Args:
            train_df: the rows the folds index into.
            folds: ``[(train_idx, val_idx), ...]``.
            val_df: the holdout frame; empty means score out-of-fold.
            hyperparameters: the caller's base hyperparameters, which a trial config overlays.
            metric: the objective key to report under.
            concurrency: how many trials run at once, for sizing per-trial CPU use.
        """
        raise NotImplementedError

    def merge_config(self, hyperparameters, config) -> dict:
        """Merge a winning search config into the base hyperparameters (phase-2 config).

        Drops the ``hpo`` block — the search is done. Override to derive knobs the search
        does not sample directly.
        """
        merged = {k: v for k, v in hyperparameters.items() if k != "hpo"}
        merged.update(config)
        return merged

    def resources_per_trial(self, hpo_block, backend):
        """Ray resource request per trial, e.g. ``{"gpu": 0.5}``. None leaves it to Ray."""
        return None


def use_holdout(requested_metric, n_val_rows: int) -> bool:
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


def run_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    adapter: HpoAdapter,
    search_space: dict,
    primary_target: str,
    output_dir: str | None = None,
) -> dict:
    """Run the search and return the hyperparameters to publish the tuned model with.

    The caller passes the already-split training frame and the held-out ``validation``
    frame (the template's ``split_validation_set`` output). Each trial trains a full
    ``n_folds`` ensemble — the same regime the winner is published in — scored on the
    held-out set (``holdout_mae``) or out-of-fold (``cv_mae``). Writes
    ``best_config.json``, ``hpo_trials.csv`` and ``hpo_rerank.csv`` to ``output_dir``
    when given.

    Cost is kept near single-fold for weak configs by reporting the running objective
    after every fold: the harness prunes a trial that is already off the pace once
    ``FOLD_PRUNE_WARMUP`` folds are in, so only promising configs pay for the full
    ensemble.

    The search does not pick the winner on its own — its finalists go through
    :func:`rerank_finalists`, which re-scores them and the *baseline* on fresh trainings
    and selects on those. The published config is whichever wins there, so a search that
    found nothing real publishes the caller's own hyperparameters unchanged. Disable with
    ``hpo["rerank_top_k"] = 0``.

    Args:
        train_df: training rows (validation rows already removed).
        val_df: the designated validation rows, or an empty frame.
        base_hyperparameters: the caller's hyperparameters, including the ``hpo`` block.
        hpo_block: the ``hpo`` block — ``n_trials``, ``backend``, ``max_parallel``,
            ``metric``, ``rerank_top_k``, ``n_folds``.
        adapter: the framework's :class:`HpoAdapter`.
        search_space: resolved ``{knob: Spec}`` for the framework.
        primary_target: the target column the objective scores.
        output_dir: where to write the HPO artifacts.

    Returns:
        dict: phase-2 hyperparameters — no ``hpo`` block.
    """
    from workbench.training.splits import get_split_indices
    from workbench.training.hpo_harness import run_search

    # The caller's holdout frame is split off *before* the template's own row filtering, so
    # both frames are filtered here rather than assumed clean.
    train_df, val_df = adapter.prepare_frame(train_df), adapter.prepare_frame(val_df)

    # The objective is the PRIMARY target's MAE, but a template keeps a row when any target
    # is non-NaN — so a multi-target frame can arrive with unlabeled primary targets. Scoring
    # nanmean-skips them; this check catches the degenerate case up front rather than after a
    # full search (a NaN objective fails every trial, which surfaces as an opaque crash).
    labeled = int(train_df[primary_target].notna().sum())
    if labeled == 0:
        raise ValueError(f"HPO objective needs non-NaN values in the primary target {primary_target!r}; found none.")
    if labeled < len(train_df):
        print(
            f"[hpo] {len(train_df) - labeled} of {len(train_df)} training rows have no {primary_target}; "
            "trained on, but excluded from scoring"
        )

    n_val_rows = len(val_df)
    if not use_holdout(hpo_block.get("metric"), n_val_rows):
        # Emptying it here is what routes the trial/re-rank scoring to out-of-fold.
        val_df = val_df.iloc[:0]

    # Same folds the template would build for this config, so a trial's ensemble matches
    # the published one. Scaffold is the SMILES default (literature-favored; random splits
    # leak near-duplicate scaffolds across train/val).
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
        test_size=0.2,
        butina_cutoff=base_hyperparameters.get("butina_cutoff", 0.4),
        **adapter.split_kwargs(),
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

    backend = hpo_block.get("backend", "auto")
    max_parallel = max(1, hpo_block.get("max_parallel", 1))
    resources = adapter.resources_per_trial(hpo_block, backend)
    print(f"[hpo] {max_parallel} concurrent trial(s), knobs={list(search_space)}")

    trial_fn = adapter.make_trial_fn(
        train_df=train_df,
        folds=folds,
        val_df=val_df,
        hyperparameters=base_hyperparameters,
        metric=metric,
        concurrency=max_parallel,
    )
    # A same-basis reference for every trial: the caller's own hyperparameters ({} overrides
    # nothing) scored on the search folds and seed. Without it the search's numbers and the
    # trials plot have nothing to anchor against, so it always runs.
    search_baseline_value = float(trial_fn({}, lambda **_: None))
    print(f"[hpo] baseline {metric}={search_baseline_value:.4f} (caller's hyperparameters, search basis)")

    result = run_search(
        trial_fn,
        search_space,
        n_trials=hpo_block.get("n_trials", 60),
        backend=backend,
        max_parallel=max_parallel,
        metric=metric,
        mode="min",
        prune_warmup=FOLD_PRUNE_WARMUP,
        seed=seed,
        resources_per_trial=resources,
    )
    pct = 100 * (search_baseline_value - result.best_value) / search_baseline_value
    print(
        f"[hpo] search best {metric}={result.best_value:.4f} ({pct:+.1f}% vs baseline, search basis)  "
        f"config={result.best_config}"
    )

    # Only completed trials are shortlist-eligible, so an unnoticed pile of failures shrinks
    # the real search budget without shrinking the reported one.
    counts = summarize_trials(result.trials)
    print(
        f"[hpo] trials: {counts['completed']} completed, {counts['pruned']} pruned, "
        f"{counts['failed']} FAILED (of {counts['attempted']})"
    )
    if counts["failed"]:
        print(
            f"[hpo] WARNING: {counts['failed']} trial(s) raised and produced no score. With trials "
            "sharing a GPU, CUDA OOM is the usual cause — check the log for OutOfMemoryError and "
            "consider hpo['gpus_per_trial']=1.0."
        )
    if counts["completed"] < max(1, counts["attempted"] // 4):
        print(
            f"[hpo] WARNING: only {counts['completed']} of {counts['attempted']} trials ran the full "
            "ensemble, so the re-rank shortlist came from a small pool — treat the margin as weak."
        )

    # Phase 1.5 refines a result the search has already produced, so its failure degrades to
    # the unrefined winner rather than discarding a search that has already run to completion.
    try:
        best_config, rerank = rerank_finalists(
            result,
            top_k=int(hpo_block.get("rerank_top_k", RERANK_TOP_K)),
            adapter=adapter,
            train_df=train_df,
            val_df=val_df,
            folds=folds,
            split_kwargs=split_kwargs,
            base_hyperparameters=base_hyperparameters,
            metric=metric,
            search_space=search_space,
            seed=seed,
            backend=backend,
            max_parallel=max_parallel,
            resources=resources,
        )
    except Exception as exc:
        print(f"[hpo] re-rank FAILED ({exc!r}); publishing the search winner unrefined")
        best_config, rerank = result.best_config, {}

    if output_dir:
        record = best_config_record(
            result,
            metric=metric,
            counts=counts,
            best_config=best_config,
            rerank=rerank,
            search_baseline_value=search_baseline_value,
        )
        with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
            json.dump(record, fp, indent=2, default=str)
        rows = trial_records(result.trials, base_hyperparameters, search_space, search_baseline_value)
        _write_records(rows, os.path.join(output_dir, "hpo_trials.csv"))
        if rerank.get("candidates"):
            _write_records(rerank["candidates"], os.path.join(output_dir, "hpo_rerank.csv"))

    return adapter.merge_config(base_hyperparameters, best_config)


def best_config_record(result, *, metric, counts, best_config, rerank, search_baseline_value) -> dict:
    """The ``best_config.json`` payload — the search's decision and what it turned on.

    Read by :func:`workbench.utils.model_utils.get_hpo_results`. The two value pairs are
    on *different bases* and must not be mixed:

    * ``best_value`` / ``baseline_value`` — the re-rank's basis, so their difference is
      the real margin the publish decision turned on. This is the pair to quote.
    * ``search_best_value`` / ``search_baseline_value`` — phase 1's own numbers, the same
      basis as every row in ``hpo_trials.csv``. When ``rerank_fresh_split`` is true the
      re-rank scored on a different fold partition, so these are not comparable to the
      pair above — partitions differ in difficulty.

    Args:
        result: the :class:`~workbench.training.hpo_harness.HpoResult` from the search.
        metric: the objective key (``cv_mae`` or ``holdout_mae``).
        counts: :func:`summarize_trials` output — completed/pruned/failed. Only
            ``completed`` trials were shortlist-eligible, so this is how much of the
            budget actually backed the result.
        best_config: the config being published.
        rerank: :func:`rerank_finalists`' info dict (empty when it was skipped).
        search_baseline_value: the caller's own hyperparameters on the search basis.

    Returns:
        dict: json-serializable record.
    """
    return {
        "metric": metric,
        "trial_counts": counts,
        "best_config": best_config,
        "best_value": rerank.get("best_value"),
        "baseline_value": rerank.get("baseline_value"),
        "search_best_value": result.best_value,
        "search_baseline_value": search_baseline_value,
        "search_best_config": result.best_config,
        "rerank_fresh_split": rerank.get("fresh_split", False),
        "rerank": rerank.get("candidates", []),
    }


def trial_records(trials, base_hyperparameters: dict, search_space: dict, baseline_value) -> list:
    """The ``hpo_trials.csv`` rows — every trial plus the baseline, one schema.

    Read by :func:`workbench.utils.model_utils.get_hpo_results`. Columns:

    * ``number`` — the trial's index; ``-1`` for the baseline row.
    * ``value`` — the objective. On a ``completed`` trial this is the full-ensemble
      score; on a pruned one it is a *partial*-ensemble score, which can read lower.
    * ``completed`` — bool, normalized across backends (Optuna reports a state name,
      Ray a flag). Pruned vs failed stays recoverable: an incomplete trial with a
      ``value`` was pruned, one without ever scored.
    * ``hyperparameters`` — every searched knob and the value it actually trained at,
      so the table is rectangular and NaN-free (see :func:`effective_config`).
    * ``kind`` — ``trial`` or ``baseline``. The baseline is the caller's own
      hyperparameters on the search basis: the reference line any plot of the search
      needs.

    Args:
        trials: the per-trial records from :class:`~workbench.training.hpo_harness.HpoResult`.
        base_hyperparameters: the caller's hyperparameters.
        search_space: the resolved ``{knob: Spec}`` space.
        baseline_value: the baseline's objective on the search basis.

    Returns:
        list: one dict per trial, with the baseline row last.
    """
    rows = [
        {
            **{k: v for k, v in trial.items() if k not in ("config", "state", "completed")},
            "completed": trial_completed(trial),
            "hyperparameters": effective_config(trial.get("config") or {}, base_hyperparameters, search_space),
            "kind": "trial",
        }
        for trial in trials
    ]
    rows.append(
        {
            "number": -1,
            "value": baseline_value,
            "completed": True,
            "hyperparameters": effective_config({}, base_hyperparameters, search_space),
            "kind": "baseline",
        }
    )
    return rows


def _json_scalar(value):
    """Unwrap a sampler's numpy scalar so json can serialize it."""
    return value.item() if hasattr(value, "item") else str(value)


def _write_records(rows, path) -> None:
    """Write search records to CSV with the ``hyperparameters`` cell as real JSON.

    A dict rendered by ``str()`` is single-quoted and not parseable by anything but
    ``ast.literal_eval``; ``json.dumps`` makes the column ``json.loads``-able, which is what
    a reader will reach for.
    """
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "hyperparameters" in frame:
        frame["hyperparameters"] = [json.dumps(h, default=_json_scalar) for h in frame["hyperparameters"]]
    frame.to_csv(path, index=False)


def effective_config(config: dict, base_hyperparameters: dict, search_space: dict) -> dict:
    """What a candidate actually trained with, for every searched knob.

    Resolution order is the same one training follows: the candidate's own override, else
    the caller's hyperparameters, else the knob's declared default. The last step is why a
    search record is always a rectangular table of real values — a knob nobody set still has
    the value it trained at, rather than a hole downstream readers have to interpret.
    """
    return {knob: config.get(knob, base_hyperparameters.get(knob, spec.default)) for knob, spec in search_space.items()}


def summarize_trials(trials) -> dict:
    """Split a search's trials into completed / pruned / failed.

    The three are not interchangeable and only ``completed`` is comparable. A *pruned*
    trial was stopped early by the scheduler and its value is a partial-ensemble score; a
    *failed* trial never produced a value at all (it raised — CUDA OOM is the usual cause
    when trials share a GPU). Failures are indistinguishable from prunes by the completion
    flag alone, which is how a run can lose a third of its budget and still look fine.
    """
    completed = [t for t in trials if trial_completed(t)]
    unfinished = [t for t in trials if not trial_completed(t)]
    return {
        "attempted": len(trials),
        "completed": len(completed),
        "pruned": len([t for t in unfinished if t.get("value") is not None]),
        "failed": len([t for t in unfinished if t.get("value") is None]),
    }


def trial_completed(trial: dict) -> bool:
    """Whether a trial ran to completion (not pruned) — both backends' record shapes."""
    if "completed" in trial:  # ray
        return bool(trial["completed"])
    return trial.get("state") == "COMPLETE"  # optuna


def shortlist_configs(trials, top_k: int) -> list:
    """The ``top_k`` distinct configs among completed trials, best (lowest) first."""
    ranked, seen = [], set()
    for trial in trials:
        value = trial.get("value")
        if value is None or not trial_completed(trial):
            continue
        config = trial.get("config") or {}
        key = json.dumps(config, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((value, config))
    ranked.sort(key=lambda pair: pair[0])
    return [config for _, config in ranked[:top_k]]


def rerank_finalists(
    result,
    *,
    top_k,
    adapter,
    train_df,
    val_df,
    folds,
    split_kwargs,
    base_hyperparameters,
    metric,
    search_space,
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
    the baseline is what gets published. Ties go to the baseline, and so does a re-rank
    whose baseline failed to score — a finalist publishes only by beating a measured
    baseline.

    Independence comes from a fresh seed (trials are deterministic, so the search seed would
    replay rather than redraw) and, in ``cv_mae`` mode, a fresh fold partition as well — the
    split is part of what the search selected against there. In ``holdout_mae`` mode the
    holdout is the user's fixed OOD set and is deliberately reused, so that pass removes the
    training-stochasticity component of the bias but not the holdout-sampling component.

    Returns:
        tuple: ``(best_config, info)`` — the config to publish, and a dict carrying
        ``candidates`` (the per-candidate record), the winning and baseline values on the
        re-rank's own basis, and ``fresh_split``. ``info`` is empty when the re-rank is
        disabled or there were no completed trials to re-rank.
    """
    from workbench.training.splits import get_split_indices
    from workbench.training.hpo_harness import evaluate_configs

    if top_k <= 0:
        print("[hpo] re-rank disabled (rerank_top_k=0); publishing the search winner")
        return result.best_config, {}

    # The baseline is the empty config: merged over base_hyperparameters it changes nothing.
    candidates = [{}] + shortlist_configs(result.trials, top_k)
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

    # At most one slot per candidate runs at a time, so per-trial CPU divides fewer ways here
    # than during the search — the adapter re-sizes rather than inheriting the tighter budget.
    rerank_parallel = min(max_parallel, len(candidates))
    rerank_fn = adapter.make_trial_fn(
        train_df=train_df,
        folds=rerank_folds,
        val_df=val_df,
        hyperparameters=base_hyperparameters,
        metric=metric,
        concurrency=rerank_parallel,
    )

    def evaluate(config, index):
        return rerank_fn({**config, "seed": rerank_seed}, lambda **_: None)

    values = evaluate_configs(
        evaluate, candidates, backend=backend, max_parallel=rerank_parallel, resources_per_trial=resources
    )
    # The shortlist is best-first, so candidate i>0 holds the search's rank-i config and the
    # label names that rank. The baseline overrides nothing, so its row is the caller's own
    # effective config.
    rows = [
        {
            "candidate": "baseline" if i == 0 else f"search_rank_{i}",
            "hyperparameters": effective_config(c, base_hyperparameters, search_space),
            metric: v,
        }
        for i, (c, v) in enumerate(zip(candidates, values))
    ]
    info = {"candidates": rows, "fresh_split": metric == "cv_mae", "baseline_value": values[0], "best_value": None}

    if values[0] is None:
        # Beating the baseline on this basis is the bar for publishing a searched config;
        # with no baseline score, no finalist can clear it — so the caller's own
        # hyperparameters ship (phase 2 trains them fresh).
        print(
            "[hpo] re-rank: the baseline failed to score, so no searched config can prove itself "
            "against it — publishing the caller's own hyperparameters"
        )
        return {}, info

    # min() over (value, index) returns the first minimum, and the baseline is index 0 — so
    # an exact tie is resolved in the baseline's favor. The baseline scored, so `scored` is
    # never empty.
    scored = [(v, i) for i, v in enumerate(values) if v is not None]
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
