"""Fast unit tests for ``workbench.training.hpo_harness`` (Optuna backend).

Pure synthetic objectives — no chemprop/GPU/AWS, so these run in the default
suite. The Ray backend needs a ray-enabled container and is not covered here.
"""

import pytest

# Workbench Imports
from workbench.training.hpo_harness import Choice, FloatRange, HpoResult, IntRange, run_search

# The Optuna backend needs the `training` extra; skip the whole module if absent.
# (hpo_harness itself imports optuna lazily, so the import above works without it.)
pytest.importorskip("optuna")


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
