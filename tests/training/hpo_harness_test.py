"""Fast unit tests for ``workbench.training.hpo_harness``, both backends.

Pure synthetic objectives — no chemprop, no GPU, no AWS — so these run in the
default suite. The Ray backend's real parallelism needs a GPU box, but its trial
plumbing runs on CPU: space translation, terminal-state reporting, winner
selection.

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
    run_search,
)


def _quadratic_objective(config, report):
    """A smooth bowl minimized at x=3.0, depth=4."""
    return (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2


SPACE = {
    "x": FloatRange(0.0, 6.0),
    "depth": IntRange(2, 6, 1),
}


def test_finds_minimum():
    """TPE converges near the known optimum (x=3, depth=4)."""
    result = run_search(_quadratic_objective, SPACE, n_trials=60, backend="optuna", metric="objective", mode="min")
    assert isinstance(result, HpoResult)
    assert result.best_value < 1.0
    assert abs(result.best_config["x"] - 3.0) < 1.0
    assert result.best_config["depth"] in (3, 4, 5)


def test_result_shape():
    """HpoResult records every trial with its resolved config."""
    result = run_search(_quadratic_objective, SPACE, n_trials=20, backend="optuna")
    assert result.metric == "objective"
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


def test_every_trial_runs_to_completion():
    """No trial is stopped early, so every value in the record is on one basis."""
    result = run_search(_quadratic_objective, SPACE, n_trials=12, backend="optuna")
    assert all(t["state"] == "COMPLETE" for t in result.trials)
    assert all(t["value"] is not None for t in result.trials)


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

    result = run_search(quadratic, SPACE, n_trials=6, backend="ray")

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

    result = run_search(oom_above_depth_3, {"depth": IntRange(2, 6)}, n_trials=8, backend="ray")

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
        metrics = {"objective": 0.25, "step": 5}

    class _Stopped:
        config = {"depth": 6}
        metrics = {"objective": 0.90, "step": 2}

    records = _resolve_trial_records([_Good(), _Dead(), _Stopped()], metric="objective", choice_options={}, max_steps=5)

    assert [r["completed"] for r in records] == [True, False, False]
    assert records[1]["value"] is None and records[1]["config"] == {}
    assert records[0]["value"] == 0.25
    # A trial stopped at a rung keeps its partial value; that is what tells it from a death.
    assert records[2]["value"] == 0.90 and records[2]["step"] == 2


def test_back_to_back_ray_searches_share_a_process(ray_cluster):
    """Each Ray entry point owns its session, so nothing outlives the run that started it.

    Tune's actor manager does not survive a second ``Tuner`` in a session it did not start —
    scheduling races the previous run's teardown and raises "Tracked actor is not managed by
    this event manager".
    """
    import ray

    from hpo_ray_trials import quadratic

    first = run_search(quadratic, SPACE, n_trials=3, backend="ray")
    assert not ray.is_initialized()

    second = run_search(quadratic, SPACE, n_trials=3, backend="ray")
    assert not ray.is_initialized()
    assert first.best_value is not None and second.best_value is not None


def test_ray_all_trials_failing_raises_actionable_error(ray_cluster):
    """No usable trial must name the problem, not die resolving a None config.

    The Optuna backend already does this; the Ray backend used to fall back to ranking
    errored trials, whose ``config`` is None, and surface an AttributeError from deep
    inside config resolution — on the GPU box that costs the most to rent.
    """
    from hpo_ray_trials import always_oom

    with pytest.raises(RuntimeError, match="no usable trial"):
        run_search(always_oom, {"depth": IntRange(2, 6)}, n_trials=2, backend="ray")


def test_is_oom_discriminates():
    """``_is_oom`` keys on torch's exception type, never on message text."""
    import torch

    from workbench.training.hpo_harness import _is_oom

    assert _is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("CUDA out of memory"))  # same words, wrong type
    assert not _is_oom(ValueError("boom"))


# --- points_to_evaluate (the baseline trial) -------------------------------


def test_a_seeded_point_is_tried_first_and_can_win():
    """The baseline rides in as an ordinary trial: the sampler sees it and it can win."""
    from workbench.training.hpo_harness import Choice, FloatRange

    space = {"x": FloatRange(0.0, 10.0, default=5.0), "shape": Choice(["a", "b", "c"], default="b")}
    seen = []

    def trial_fn(config, report):
        seen.append(config)
        # Only the seeded point scores 0.0, so it wins unless it was never tried.
        return 0.0 if (config["x"], config["shape"]) == (5.0, "b") else 1.0

    result = run_search(
        trial_fn, space, n_trials=5, backend="optuna", mode="min", points_to_evaluate=[{"x": 5.0, "shape": "b"}]
    )
    assert seen[0] == {"x": 5.0, "shape": "b"}  # tried before anything the sampler proposes
    assert result.best_config == {"x": 5.0, "shape": "b"} and result.best_value == 0.0


def test_a_seeded_choice_outside_the_space_is_rejected():
    """A Choice is sampled as an index, so a value the space lacks cannot be expressed."""
    import pytest

    from workbench.training.hpo_harness import Choice

    with pytest.raises(ValueError, match="not one of the searched options"):
        run_search(
            lambda c, report: 1.0,
            {"shape": Choice(["a", "b"], default="a")},
            n_trials=2,
            backend="optuna",
            points_to_evaluate=[{"shape": "z"}],
        )


# --- the fold ladder (successive halving over reported steps) --------------


def _laddered_search(n_trials=20, seeded=None, max_steps=5):
    """A search whose objective is `x` at every step, so rung decisions are unambiguous.

    Returns ``(result, steps_run)`` — ``steps_run`` maps each trial's x to how many steps it
    got through, which is what says whether the scheduler stopped it.
    """
    from workbench.training.hpo_harness import FloatRange

    steps_run = {}

    def trial_fn(config, report):
        x = config["x"]
        for step in range(1, max_steps + 1):
            steps_run[x] = step
            report(step, x)
        return x

    result = run_search(
        trial_fn,
        {"x": FloatRange(0.0, 10.0, default=5.0)},
        n_trials=n_trials,
        backend="optuna",
        mode="min",
        max_steps=max_steps,
        points_to_evaluate=seeded,
    )
    return result, steps_run


def test_the_ladder_stops_trials_before_their_last_step():
    """Successive halving has to actually cull, or the ladder buys nothing."""
    result, steps_run = _laddered_search()

    stopped = [t for t in result.trials if t["state"] == "PRUNED"]
    assert stopped, "no trial was stopped early — the pruner is not engaged"
    # A stopped trial keeps the partial value it did report; that is what tells it from a death.
    assert all(t["value"] is not None and t["step"] < 5 for t in stopped)
    # And the total work is less than running everything to term.
    assert sum(steps_run.values()) < 5 * len(steps_run)


def test_the_ladder_only_ranks_trials_that_ran_every_step():
    """A partial objective covers fewer steps, so it must not win."""
    result, _ = _laddered_search()

    completed = [t for t in result.trials if t["state"] == "COMPLETE"]
    assert result.best_value == min(t["value"] for t in completed)
    assert result.best_config["x"] == result.best_value


def test_a_seeded_point_is_never_stopped_early():
    """The baseline reports nothing until the end, so no rung ever sees it.

    Seeded deliberately bad (x=9.5): anything prunable at that value dies at the first rung,
    so reaching step 5 can only mean the scheduler never got a look at it.
    """
    result, steps_run = _laddered_search(seeded=[{"x": 9.5}])

    assert steps_run[9.5] == 5
    baseline = next(t for t in result.trials if t["config"].get("x") == 9.5)
    assert baseline["state"] == "COMPLETE" and baseline["value"] == 9.5


def test_a_nan_objective_after_reporting_is_failed_not_stopped():
    """Optuna marks a NaN-returning trial FAILED, but it may already have reported
    intermediates. Backfilling its value from those would file a genuine failure as a
    scheduler stop, hiding the one count that means the budget was lost."""
    from workbench.training.hpo_harness import FloatRange
    from workbench.training.hpo_runner import summarize_trials

    max_steps = 4

    def trial_fn(config, report):
        report(1, config["x"])  # reports before failing
        if config["x"] > 5.0:
            return float("nan")
        for step in range(2, max_steps + 1):
            report(step, config["x"])
        return config["x"]

    result = run_search(
        trial_fn,
        {"x": FloatRange(0.0, 10.0, default=5.0)},
        n_trials=12,
        backend="optuna",
        mode="min",
        max_steps=max_steps,
    )
    died = [t for t in result.trials if t["state"] == "FAIL"]
    assert died, "expected some trials to produce a NaN objective"
    assert all(t["value"] is None for t in died), "a failed trial must not carry a partial value"
    counts = summarize_trials(result.trials)
    assert counts["failed"] == len(died)
