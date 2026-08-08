"""Framework-agnostic hyperparameter-search orchestration.

Sits between :mod:`workbench.training.hpo_harness` (the sampler/backend layer) and a
framework module (:mod:`workbench.training.chemprop_hpo`, :mod:`workbench.training.xgb_hpo`).
Everything a search does *around* training a candidate lives here: fold construction, the
baseline trial, and the ``best_config.json`` / ``hpo_trials.csv`` artifacts that
:func:`workbench.utils.training_job_utils.get_hpo_results` reads.

A framework supplies an :class:`HpoAdapter` — how to train and score one candidate, and how
a winning config merges back into the published hyperparameters.

The values a search reports are *search diagnostics*, not performance estimates: the
winner is the minimum over many noisy evaluations, so its value is optimistically biased,
and every candidate was scored on the data that selected it. What the published model is
actually worth comes from the phase-2 metrics and whatever holdout or champion/challenger
comparison the caller runs afterwards.

Training-only; imported from inside a model template's ``__main__``.
"""

from __future__ import annotations

import json
import os

# The objective, always: pooled out-of-fold MAE on the primary target. Designated validation
# rows are never scored during a search — they are the only unbiased read on the published
# model, and a few hundred of them make a noisier selection signal than thousands of
# out-of-fold predictions.
METRIC = "cv_mae"


class HpoAdapter:
    """Framework hooks for :func:`run_hpo`.

    A subclass carries whatever the framework needs to train a candidate (target columns,
    feature list, model shape) and implements :meth:`make_trial_fn`. The rest have working
    defaults.
    """

    # Search budget when the caller sets no ``hpo["n_trials"]``. Sized against what one
    # trial costs: an XGBoost fit is seconds, a chemprop ensemble is hours. A subclass that
    # says nothing gets a middling budget rather than a wrong one.
    default_n_trials = 60

    def split_kwargs(self) -> dict:
        """Extra ``get_split_indices`` kwargs, e.g. the SMILES column for scaffold folds."""
        return {}

    def make_trial_fn(self, *, train_df, folds, hyperparameters, concurrency):
        """Build the ``(config, report) -> float`` objective for the search.

        A trial trains the full ensemble the winner is published as and returns its pooled
        out-of-fold MAE, calling ``report(fold, running_mae)`` after each member so the
        harness can stop a trial that is already off the pace.

        Args:
            train_df: the rows the folds index into.
            folds: ``[(train_idx, val_idx), ...]``.
            hyperparameters: the caller's base hyperparameters, which a trial config overlays.
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


def visible_gpus() -> int:
    """GPUs the training container can actually see. 0 when torch is absent or CPU-only."""
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count()


def resolve_max_parallel(hpo_block, resources, n_gpus: int) -> int:
    """How many trials to run at once, derived from the GPUs actually present.

    The adapter's ``resources_per_trial`` already sets a trial's GPU share, so the cards on
    the box divide by that share to give the concurrency they support. Deriving it keeps the
    packing decision in one place: a hand-set ``max_parallel`` can contradict the share, and
    only the container knows which rung of the instance ladder capacity actually granted.

    Args:
        hpo_block: the ``hpo`` block; an explicit ``max_parallel`` overrides the derived value.
        resources: the adapter's per-trial resource request, or None.
        n_gpus: GPUs visible to this process.

    Returns:
        int: concurrent trials, at least 1.
    """
    requested = hpo_block.get("max_parallel")
    if requested is not None:
        return max(1, int(requested))
    share = (resources or {}).get("gpu")
    if not share or not n_gpus:
        return 1
    return max(1, int(n_gpus / share))


def run_hpo(
    train_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    adapter: HpoAdapter,
    search_space: dict,
    primary_target: str,
    output_dir: str | None = None,
) -> dict:
    """Run the search and return the hyperparameters to publish the tuned model with.

    Each trial trains a full ``n_folds`` ensemble — the same regime the winner is published
    in — scored as pooled out-of-fold MAE on ``primary_target``. Every trial is on that one
    basis, so every value in the record is directly comparable to every other. Writes
    ``best_config.json`` and ``hpo_trials.csv`` to ``output_dir`` when given.

    The caller's own hyperparameters run as the first trial. That gives the record a
    reference line — the value an untuned model would have scored on these folds — and gives
    the sampler an anchored observation to start from. It can win, in which case the search
    publishes the caller's config unchanged.

    Args:
        train_df: training rows; designated validation rows are already removed.
        base_hyperparameters: the caller's hyperparameters, including the ``hpo`` block.
        hpo_block: the ``hpo`` block — ``n_trials``, ``backend``, and the packing overrides
            ``max_parallel`` / ``gpus_per_trial``, both derived from the box when unset.
        adapter: the framework's :class:`HpoAdapter`.
        search_space: resolved ``{knob: Spec}`` for the framework.
        primary_target: the target column the objective scores.
        output_dir: where to write the HPO artifacts.

    Returns:
        dict: phase-2 hyperparameters — no ``hpo`` block.
    """
    from workbench.training.hpo_harness import run_search
    from workbench.training.splits import get_split_indices

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

    # Same folds the template would build for this config, so a trial's ensemble matches the
    # published one. Scaffold is the SMILES default (literature-favored; random splits leak
    # near-duplicate scaffolds across train/val).
    n_folds = int(base_hyperparameters.get("n_folds", 5))
    strategy = base_hyperparameters.get("split_strategy", "scaffold")
    seed = base_hyperparameters.get("seed", 42)
    folds = get_split_indices(
        train_df,
        random_state=seed,
        n_splits=n_folds,
        strategy=strategy,
        test_size=0.2,
        butina_cutoff=base_hyperparameters.get("butina_cutoff", 0.4),
        **adapter.split_kwargs(),
    )
    print(
        f"[hpo] objective = {METRIC} on out-of-fold {strategy} splits; "
        f"{n_folds}-fold ensemble per trial, {len(train_df)} training rows"
    )

    backend = hpo_block.get("backend", "auto")
    resources = adapter.resources_per_trial(hpo_block, backend)
    n_gpus = visible_gpus()
    n_trials = int(hpo_block.get("n_trials", adapter.default_n_trials))
    max_parallel = resolve_max_parallel(hpo_block, resources, n_gpus)
    print(f"[hpo] {max_parallel} concurrent trial(s) on {n_gpus} GPU(s), knobs={list(search_space)}")
    if max_parallel == 1:
        # Serial turns a multi-hour search into a multiple of that. Say so up front rather
        # than leaving it to be inferred from a job that ran long or hit its timeout.
        print(f"[hpo] WARNING: {n_trials} trials will run one at a time; expect a long job.")

    trial_fn = adapter.make_trial_fn(
        train_df=train_df,
        folds=folds,
        hyperparameters=base_hyperparameters,
        concurrency=max_parallel,
    )
    result = run_search(
        trial_fn,
        search_space,
        n_trials=n_trials,
        backend=backend,
        max_parallel=max_parallel,
        metric=METRIC,
        mode="min",
        seed=seed,
        resources_per_trial=resources,
        points_to_evaluate=[effective_config({}, base_hyperparameters, search_space)],
        max_steps=n_folds,
    )

    rows = trial_records(result.trials, base_hyperparameters, search_space)
    baseline = baseline_value(rows)
    counts = summarize_trials(result.trials)
    print(
        f"[hpo] trials: {counts['completed']} ran all {n_folds} folds, {counts['pruned']} stopped early, "
        f"{counts['failed']} FAILED (of {counts['attempted']})"
    )
    if counts["failed"]:
        print(
            f"[hpo] WARNING: {counts['failed']} trial(s) raised and produced no score. With trials "
            "sharing a GPU, CUDA OOM is the usual cause — check the log for OutOfMemoryError and "
            "consider hpo['gpus_per_trial']=1.0."
        )
    margin = f" ({100 * (baseline - result.best_value) / baseline:+.1f}% vs baseline)" if baseline else ""
    print(f"[hpo] search best {METRIC}={result.best_value:.4f}{margin}  config={result.best_config}")
    # The winner is the minimum over every trial, so its value is the luckiest draw of many
    # and overstates what the config is worth. Only a measurement the search did not select
    # on — the phase-2 metrics, a holdout, a champion/challenger run — settles that.
    print("[hpo] NOTE: the search's own margin is optimistic; confirm against a measurement it did not select on.")

    if output_dir:
        record = best_config_record(result, counts=counts, baseline=baseline)
        with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
            json.dump(record, fp, indent=2, default=str)
        _write_records(rows, os.path.join(output_dir, "hpo_trials.csv"))

    return adapter.merge_config(base_hyperparameters, result.best_config)


def best_config_record(result, *, counts, baseline) -> dict:
    """The ``best_config.json`` payload — the search's decision and what it turned on.

    Read by :func:`workbench.utils.training_job_utils.get_hpo_results`. The ``search_``
    prefixes are deliberate: these are the search's own numbers, on the folds it selected
    against, and they are not an estimate of what the published model is worth.

    Args:
        result: the :class:`~workbench.training.hpo_harness.HpoResult` from the search.
        counts: :func:`summarize_trials` output — completed/pruned/failed. Only completed
            trials could win, so this is how much of the budget actually backed the result.
        baseline: the caller's own hyperparameters as scored by their trial, or None when
            that trial failed.

    Returns:
        dict: json-serializable record.
    """
    return {
        "metric": METRIC,
        "trial_counts": counts,
        "best_config": result.best_config,
        "search_best_value": result.best_value,
        "search_baseline_value": baseline,
    }


def trial_records(trials, base_hyperparameters: dict, search_space: dict) -> list:
    """The ``hpo_trials.csv`` rows — one schema for every trial.

    Read by :func:`workbench.utils.training_job_utils.get_hpo_results`. Columns:

    * ``number`` — the trial's index.
    * ``value`` — the objective, or empty for a trial that died before scoring.
    * ``completed`` — bool, normalized across backends (Optuna reports a state name,
      Ray a flag). True only for a trial that ran every fold; a trial stopped at a rung
      keeps its partial ``value`` but is not completed.
    * ``step`` — the fold it last reported at, so a stopped trial says where it stopped.
    * ``hyperparameters`` — every searched knob and the value it actually trained at,
      so the table is rectangular and NaN-free (see :func:`effective_config`).
    * ``kind`` — ``baseline`` for the trial that trained at the caller's own settings,
      else ``trial``. The baseline is the reference line any plot of the search needs.
    """
    baseline = effective_config({}, base_hyperparameters, search_space)
    rows, seen_baseline = [], False
    for trial in trials:
        effective = effective_config(trial.get("config") or {}, base_hyperparameters, search_space)
        # First match only. The seeded point runs first, but a sampler can land on the same
        # config again on a discrete space — and a second `baseline` row would drop a real
        # trial out of the plots and make the reference line ambiguous.
        is_baseline = not seen_baseline and effective == baseline
        seen_baseline = seen_baseline or is_baseline
        rows.append(
            {
                **{k: v for k, v in trial.items() if k not in ("config", "state", "completed")},
                "completed": trial_completed(trial),
                "hyperparameters": effective,
                "kind": "baseline" if is_baseline else "trial",
            }
        )
    return rows


def baseline_value(rows) -> "float | None":
    """The baseline trial's objective, or None when it never scored.

    The baseline is enqueued as a search point rather than run separately, so it is found in
    the records rather than held aside. Nothing downstream requires it — a failed baseline
    costs the plots their reference line, not the search its winner.
    """
    return next((r["value"] for r in rows if r["kind"] == "baseline" and r["value"] is not None), None)


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


def pooled_mae(oof_pred, oof_true) -> float:
    """MAE over every out-of-fold prediction made so far.

    Pooled, not a mean of per-fold means: every row weighs the same, which is what the
    model's own cross-fold metrics report. Fold sizes differ under a scaffold split, so the
    two are not the same number.

    nanmean, not mean: the template keeps a row when ANY target is non-NaN, so a
    multi-target frame can carry a NaN primary target. Training still uses every row
    (chemprop masks per-target); only the scoring skips the unlabeled ones.

    Returns NaN when the folds so far hold no labelled primary target at all — possible on
    sparse multi-target data. Callers must not report that as a rung: it is an absence of
    measurement, not a bad one.
    """
    import numpy as np

    error = np.abs(np.concatenate(oof_pred) - np.concatenate(oof_true))
    if not np.isfinite(error).any():
        return float("nan")
    return float(np.nanmean(error))


def effective_config(config: dict, base_hyperparameters: dict, search_space: dict) -> dict:
    """What a candidate actually trained with, for every searched knob.

    Resolution order is the same one training follows: the candidate's own override, else
    the caller's hyperparameters, else the knob's declared default. The last step is why a
    search record is always a rectangular table of real values — a knob nobody set still has
    the value it trained at, rather than a hole downstream readers have to interpret.
    """
    return {knob: config.get(knob, base_hyperparameters.get(knob, spec.default)) for knob, spec in search_space.items()}


def summarize_trials(trials) -> dict:
    """Count a search's trials by outcome. The three are not interchangeable.

    * ``completed`` — ran every fold, so its objective is comparable and it could win.
    * ``pruned`` — stopped at a rung. It has a value, but over fewer folds than a full
      trial, so it is a record of where the search looked rather than a ranking entry.
    * ``failed`` — produced no value at all (it raised; CUDA OOM is the usual cause when
      trials share a GPU). This is how a run loses a chunk of its budget while still
      looking fine from the trial count alone.
    """
    completed = [t for t in trials if trial_completed(t)]
    scored = [t for t in trials if t.get("value") is not None]
    return {
        "attempted": len(trials),
        "completed": len(completed),
        "pruned": len(scored) - len(completed),
        "failed": len(trials) - len(scored),
    }


def trial_completed(trial: dict) -> bool:
    """Whether a trial produced a usable objective — both backends' record shapes."""
    if "completed" in trial:  # ray
        return bool(trial["completed"])
    return trial.get("state") == "COMPLETE"  # optuna
