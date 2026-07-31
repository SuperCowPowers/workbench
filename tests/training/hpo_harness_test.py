"""Fast unit tests for ``workbench.training.hpo_harness``, both backends.

Pure synthetic objectives — no chemprop, no GPU, no AWS — so these run in the
default suite. The Ray backend's real parallelism needs a GPU box, but its trial
plumbing runs on CPU: space translation, terminal-state reporting, rung layout,
winner selection.

optuna, ray and torch all arrive with the `modeling` extra, which every tox test
env installs — a missing one is a broken environment, not a reason to skip.
"""

import pytest

# Workbench Imports
from workbench.training.hpo_harness import (
    Choice,
    FloatRange,
    HpoResult,
    IntRange,
    evaluate_configs,
    run_search,
)


def _quadratic_objective(config, report):
    """A smooth bowl minimized at x=3.0, depth=4; reports one intermediate step."""
    value = (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2
    report(step=1, holdout_mae=value)  # exercise the report/prune path
    return value


SPACE = {
    "x": FloatRange(0.0, 6.0),
    "depth": IntRange(2, 6, 1),
}


def test_finds_minimum():
    """TPE converges near the known optimum (x=3, depth=4)."""
    result = run_search(_quadratic_objective, SPACE, n_trials=60, backend="optuna", metric="holdout_mae", mode="min")
    assert isinstance(result, HpoResult)
    assert result.best_value < 1.0
    assert abs(result.best_config["x"] - 3.0) < 1.0
    assert result.best_config["depth"] in (3, 4, 5)


def test_result_shape():
    """HpoResult records every trial with its resolved config."""
    result = run_search(_quadratic_objective, SPACE, n_trials=20, backend="optuna")
    assert result.metric == "holdout_mae"
    assert result.mode == "min"
    assert result.n_trials == 20
    assert len(result.trials) == 20
    t0 = result.trials[0]
    assert set(t0["config"]) == {"x", "depth"}
    assert isinstance(t0["config"]["depth"], int)


def test_choice_with_list_options():
    """Unhashable Choice options (a tapered ffn head) resolve to real values."""
    options = [2000, 1000, [1024, 256, 64]]

    def obj(config, report):
        ffn = config["ffn_hidden_dim"]
        # Prefer the tapered list; scalars score worse.
        return 0.0 if isinstance(ffn, list) else float(ffn)

    space = {"ffn_hidden_dim": Choice(options)}
    result = run_search(obj, space, n_trials=25, backend="optuna")
    assert result.best_config["ffn_hidden_dim"] in options
    assert result.best_config["ffn_hidden_dim"] == [1024, 256, 64]


def test_maximize_mode():
    """mode='max' flips the optimization direction."""

    def obj(config, report):
        return -((config["x"] - 2.0) ** 2)  # peak at x=2

    space = {"x": FloatRange(0.0, 4.0)}
    result = run_search(obj, space, n_trials=40, backend="optuna", mode="max")
    assert abs(result.best_config["x"] - 2.0) < 1.0
    assert result.best_value > -1.0


def test_invalid_mode_raises():
    """A bad mode fails loudly rather than silently optimizing the wrong direction."""
    with pytest.raises(ValueError, match="mode must be"):
        run_search(_quadratic_objective, SPACE, n_trials=1, backend="optuna", mode="minimum")


def test_invalid_backend_raises():
    """A typo'd backend fails loudly rather than silently falling back to optuna."""
    with pytest.raises(ValueError, match="backend must be"):
        run_search(_quadratic_objective, SPACE, n_trials=1, backend="optuaa")


def test_default_pruning_respects_warmup():
    """With pruning on, trials reporting fewer steps than the warmup are never pruned.

    Guards the grace period: an over-eager pruner kills configs before they're rankable
    (which collapsed a real 10-trial search down to 2 usable evaluations).
    """
    result = run_search(_quadratic_objective, SPACE, n_trials=12, backend="optuna")
    assert all(t["state"] == "COMPLETE" for t in result.trials)


def test_pruning_disabled_runs_all_trials():
    """With pruning off, every trial completes (none pruned)."""
    result = run_search(_quadratic_objective, SPACE, n_trials=15, backend="optuna", pruning=False)
    assert all(t["state"] == "COMPLETE" for t in result.trials)


def _multistep_objective(config, report):
    """Reports 10 intermediate steps so a warmup window has room to apply."""
    err = abs(config["x"] - 3.0)
    for step in range(1, 11):
        report(step=step, holdout_mae=err)
    return err


def test_prune_warmup_above_reported_steps_disables_pruning():
    """A non-default prune_warmup beyond every trial's reported steps makes nothing prunable.

    Exercises that prune_warmup is actually threaded (not just the module default): trials
    report 10 steps, warmup is 50, so no trial ever becomes eligible and all complete.
    """
    result = run_search(
        _multistep_objective, {"x": FloatRange(0.0, 6.0)}, n_trials=20, backend="optuna", prune_warmup=50
    )
    assert all(t["state"] == "COMPLETE" for t in result.trials)


# --- evaluate_configs (the re-rank primitive) ------------------------------


def test_evaluate_configs_scores_every_config_in_order():
    """Values come back positionally aligned with the configs, one call each."""
    seen = []

    def eval_fn(config, index):
        seen.append(index)
        return config["x"] * 2

    values = evaluate_configs(eval_fn, [{"x": 1}, {"x": 2}, {"x": 3}], backend="optuna")
    assert values == [2, 4, 6]
    assert seen == [0, 1, 2]


def test_evaluate_configs_handles_unhashable_config_values():
    """Configs may hold tapered lists — nothing here hashes them."""
    configs = [{"ffn_hidden_dim": [1024, 256, 64]}, {"ffn_hidden_dim": 600}]
    values = evaluate_configs(lambda c, i: float(i), configs, backend="optuna")
    assert values == [0.0, 1.0]


def test_evaluate_configs_isolates_failures():
    """One config blowing up yields None for that slot, not a lost run."""

    def eval_fn(config, index):
        if index == 1:
            raise RuntimeError("boom")
        return 1.0

    assert evaluate_configs(eval_fn, [{}, {}, {}], backend="optuna") == [1.0, None, 1.0]


def test_evaluate_configs_empty():
    assert evaluate_configs(lambda c, i: 1.0, [], backend="optuna") == []


def test_all_nan_objective_raises_actionable_error():
    """A NaN objective fails every Optuna trial — the error must say why, not 'no trials'.

    A multi-task frame whose primary target is unlabeled on some rows can drive the objective
    to NaN, which Optuna records as a failed trial. The whole search is spent by the time this
    surfaces, so the message has to name the cause.
    """

    def nan_objective(config, report):
        return float("nan")

    with pytest.raises(RuntimeError, match="no usable trial"):
        run_search(nan_objective, {"x": FloatRange(0.0, 1.0)}, n_trials=3, backend="optuna")


def test_partial_nan_objective_still_finds_the_best():
    """Some trials NaN-ing out doesn't sink the search — the scorable ones still rank."""

    def sometimes_nan(config, report):
        return config["x"] if config["x"] < 0.5 else float("nan")

    result = run_search(sometimes_nan, {"x": FloatRange(0.0, 1.0)}, n_trials=25, backend="optuna")
    assert result.best_value < 0.5


def test_gpu_fence_is_a_noop_without_a_gpu_allocation():
    """A CPU-resourced trial (or no resource request at all) must not touch torch."""
    from workbench.training.hpo_harness import _fence_gpu_memory

    _fence_gpu_memory(None)
    _fence_gpu_memory({})
    _fence_gpu_memory({"cpu": 4})  # the XGBoost-on-ray shape


def test_fence_leaves_headroom_for_co_tenant_cuda_contexts():
    """Shares that sum to a whole card must still fence below it.

    ``set_per_process_memory_fraction`` is a fraction of *total* memory, so two trials
    fenced at a literal 0.5 leave nothing for their CUDA contexts — which the driver
    allocates outside the caching allocator.
    """
    from workbench.training.hpo_harness import _FENCE_HEADROOM

    assert _FENCE_HEADROOM < 1.0
    assert 2 * (0.5 * _FENCE_HEADROOM) < 1.0


# --- Ray backend ------------------------------------------------------------
# Ray is a test dependency (via the `training` extra, which `test` pulls in), so these
# run in the default suite. They cover the trial plumbing — space translation, terminal
# state reporting, winner selection — on CPU. Real parallelism still needs a GPU box.


def test_ray_space_maps_choices_to_indices():
    """``Choice`` knobs sample as an index so unhashable options survive Optuna."""
    from workbench.training.hpo_harness import _to_ray_space

    space, options = _to_ray_space(
        {"width": IntRange(2, 10, 2), "shape": Choice([[1, 2], [3, 4]]), "lr": FloatRange(1e-4, 1e-1, log=True)}
    )
    assert options["shape"] == [[1, 2], [3, 4]]
    assert "width" not in options and "lr" not in options
    assert set(space) == {"width", "shape", "lr"}


def test_ray_search_finds_minimum(ray_cluster):
    """End-to-end on the Ray backend: the reported winner is the sampled optimum."""
    from hpo_ray_trials import quadratic

    result = run_search(quadratic, SPACE, n_trials=6, backend="ray", pruning=False)

    assert isinstance(result, HpoResult)
    assert result.best_value == pytest.approx(min(t["value"] for t in result.trials))
    assert set(result.best_config) == set(SPACE)


def test_ray_oom_trial_is_scored_none_rather_than_erroring(ray_cluster):
    """An out-of-memory trial reports a null objective instead of dying.

    Ray tells Optuna FAIL for an *errored* trial, and TPE draws only on COMPLETE/PRUNED —
    so a crashed trial teaches the sampler nothing and the corner gets re-proposed for the
    rest of the search. A null objective lands it in the pruned bucket, which TPE models.
    """
    from hpo_ray_trials import oom_above_depth_3

    result = run_search(oom_above_depth_3, {"depth": IntRange(2, 6)}, n_trials=8, backend="ray", pruning=False)

    oomed = [t for t in result.trials if t["config"]["depth"] >= 4]
    assert oomed, "search never sampled the failing region"
    assert all(t["value"] is None for t in oomed)
    assert all(not t["completed"] for t in oomed)
    # The winner still comes from the trials that actually ran.
    assert result.best_value is not None
    assert result.best_config["depth"] < 4


def test_a_trial_that_died_before_reporting_does_not_sink_the_record():
    """A killed worker records no config; the other trials' results must still come back.

    Ray leaves ``config``/``metrics`` unset when a trial dies before it runs — an OOM-killed
    worker, a missing dep in the image, actor construction failing. Reading those unguarded
    raised an AttributeError from config resolution, discarding a search that had already
    been paid for in full.
    """
    from workbench.training.hpo_harness import _resolve_trial_records

    class _Dead:
        config = None
        metrics = None

    class _Good:
        config = {"depth": 3}
        metrics = {"holdout_mae": 0.25, "_hpo_completed": 1}

    records = _resolve_trial_records(
        [_Good(), _Dead(), _Good()], metric="holdout_mae", done_flag="_hpo_completed", choice_options={}
    )

    assert [r["completed"] for r in records] == [True, False, True]
    assert records[1]["value"] is None
    assert records[1]["config"] == {}
    assert records[0]["value"] == 0.25


def test_asha_gives_a_fold_search_more_than_one_rung():
    """A fold-reporting search must get a second look before a config is dropped.

    Rungs sit at ``grace_period * reduction_factor ** k``. Under Ray's default factor of 4
    the second rung lands at 8 — past the end of a 5-fold run — so the whole search turns on
    one comparison made after two folds. Reads Ray's bracket internals deliberately: the
    rung ladder is the behavior under test, and it is not otherwise observable.
    """
    from ray.tune.schedulers import ASHAScheduler

    from workbench.training.hpo_harness import PRUNE_REDUCTION_FACTOR

    scheduler = ASHAScheduler(
        metric="holdout_mae",
        mode="min",
        time_attr="step",
        grace_period=2,
        reduction_factor=PRUNE_REDUCTION_FACTOR,
    )
    rungs = sorted(rung for rung, _ in scheduler._brackets[0]._rungs)

    assert len([r for r in rungs if r <= 5]) >= 2, f"a 5-fold search sees only rungs {rungs}"


def test_partial_trials_reach_optuna_as_pruned_not_complete():
    """A scheduler-stopped trial must not enter TPE's fit as a completed observation.

    Its last value is a partial ensemble — worse than a full one by construction — so
    recording it as COMPLETE mixes two objectives in the quantile split and labels every
    pruned region worse than it is.
    """
    import optuna

    from workbench.training.hpo_harness import _partial_aware_search

    from ray import tune

    search = _partial_aware_search(
        "_hpo_completed", space={"x": tune.uniform(0.0, 1.0)}, metric="holdout_mae", mode="min", seed=42
    )

    finished, stopped = "trial_finished", "trial_stopped"
    for trial_id in (finished, stopped):
        search.suggest(trial_id)

    search.on_trial_complete(finished, result={"holdout_mae": 0.5, "_hpo_completed": 1})
    search.on_trial_complete(stopped, result={"holdout_mae": 0.4})  # better-looking, but partial

    states = {t.state for t in search._ot_study.trials}
    assert optuna.trial.TrialState.COMPLETE in states
    assert optuna.trial.TrialState.PRUNED in states

    completed = [t for t in search._ot_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 1
    assert completed[0].value == pytest.approx(0.5)
    # The partial trial's flattering 0.4 must not be sitting in the completed pool.
    assert all(t.value != pytest.approx(0.4) for t in completed)


def test_ray_all_trials_failing_raises_actionable_error(ray_cluster):
    """No usable trial must name the problem, not die resolving a None config.

    The Optuna backend already does this; the Ray backend used to fall back to ranking
    errored trials, whose ``config`` is None, and surface an AttributeError from deep
    inside config resolution — on the GPU box that costs the most to rent.
    """
    from hpo_ray_trials import always_oom

    with pytest.raises(RuntimeError, match="no usable trial"):
        run_search(always_oom, {"depth": IntRange(2, 6)}, n_trials=2, backend="ray", pruning=False)


def test_is_oom_discriminates():
    """``_is_oom`` keys on torch's exception type, never on message text."""
    import torch

    from workbench.training.hpo_harness import _is_oom

    assert _is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("CUDA out of memory"))  # same words, wrong type
    assert not _is_oom(ValueError("boom"))
