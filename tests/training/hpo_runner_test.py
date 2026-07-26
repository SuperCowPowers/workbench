"""Fast unit tests for the framework-agnostic pieces of ``workbench.training.hpo_runner``.

Objective selection, shortlisting, and the default config merge — no framework, no
training. The per-framework objectives are covered by ``chemprop_hpo_test.py`` and
``xgb_hpo_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_runner import HpoAdapter, shortlist_configs, trial_completed, use_holdout


def test_split_kwargs_default_is_empty():
    """The base adapter adds nothing to get_split_indices — frameworks with a molecule
    column (chemprop) override to name it."""
    assert HpoAdapter().split_kwargs() == {}


def test_trial_completed_reads_both_backend_shapes():
    """Ray records a `completed` flag; Optuna records a state name."""
    assert trial_completed({"completed": True}) is True
    assert trial_completed({"completed": False}) is False
    assert trial_completed({"state": "COMPLETE"}) is True
    assert trial_completed({"state": "PRUNED"}) is False
    assert trial_completed({}) is False  # neither shape → not completed


def test_shortlist_ranks_completed_trials_best_first():
    """Pruned/unscored trials are excluded and the rest sort by objective."""
    trials = [
        {"value": 0.9, "state": "COMPLETE", "config": {"depth": 2}},
        {"value": 0.5, "state": "COMPLETE", "config": {"depth": 3}},
        {"value": 0.1, "state": "PRUNED", "config": {"depth": 4}},  # pruned value is partial
        {"value": None, "state": "COMPLETE", "config": {"depth": 5}},
        {"value": 0.7, "state": "COMPLETE", "config": {"depth": 6}},
    ]
    assert shortlist_configs(trials, 3) == [{"depth": 3}, {"depth": 6}, {"depth": 2}]
    assert shortlist_configs(trials, 1) == [{"depth": 3}]


def test_shortlist_dedupes_and_handles_unhashable_configs():
    """A repeated config is listed once, including configs holding list-valued knobs."""
    tapered = {"ffn_hidden_dim": [1024, 256, 64]}
    trials = [
        {"value": 0.4, "completed": True, "config": tapered},
        {"value": 0.6, "completed": True, "config": dict(tapered)},  # same config, worse draw
        {"value": 0.5, "completed": True, "config": {"ffn_hidden_dim": 600}},
    ]
    assert shortlist_configs(trials, 5) == [tapered, {"ffn_hidden_dim": 600}]


def test_shortlist_empty_when_everything_pruned():
    """Nothing completed → no finalists (caller falls back to the search winner)."""
    assert shortlist_configs([{"value": 0.3, "completed": False, "config": {"depth": 2}}], 5) == []


def test_use_holdout_defaults_to_out_of_fold():
    """Default never tunes on the validation rows — a benchmark holdout stays uncontaminated."""
    assert use_holdout(None, 500) is False
    assert use_holdout(None, 0) is False


def test_cv_mae_override_ignores_the_holdout():
    """Asking for cv_mae explicitly matches the default."""
    assert use_holdout("cv_mae", 500) is False
    assert use_holdout("cv_mae", 0) is False


def test_holdout_mae_is_opt_in_and_requires_validation_rows():
    """Tuning toward the holdout happens only when asked for, and only if rows exist."""
    assert use_holdout("holdout_mae", 500) is True
    try:
        use_holdout("holdout_mae", 0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "validation_ids" in str(exc)


def test_unknown_metric_raises():
    """A typo'd metric fails rather than silently falling back to a default objective."""
    try:
        use_holdout("holdout_rmse", 500)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must be" in str(exc)


def test_default_merge_drops_hpo_block_and_applies_winner():
    """The base adapter merge overlays the winner and strips the hpo block."""
    merged = HpoAdapter().merge_config({"max_depth": 7, "hpo": {"n_trials": 40}}, {"max_depth": 3})
    assert merged == {"max_depth": 3}


def test_default_adapter_passes_frames_through():
    """A framework with nothing to filter inherits an identity prepare_frame."""
    frame = object()
    assert HpoAdapter().prepare_frame(frame) is frame
    assert HpoAdapter().resources_per_trial({}, "ray") is None


def test_summarize_trials_separates_pruned_from_failed():
    """A pruned trial has a partial value; a failed one has none. Both are 'not completed'."""
    from workbench.training.hpo_runner import summarize_trials

    counts = summarize_trials(
        [
            {"completed": True, "value": 0.5},
            {"completed": True, "value": 0.6},
            {"completed": False, "value": 0.9},  # ASHA-stopped, partial ensemble
            {"completed": False, "value": None},  # raised (e.g. CUDA OOM)
            {"completed": False, "value": None},
        ]
    )
    assert counts == {"attempted": 5, "completed": 2, "pruned": 1, "failed": 2}


def test_summarize_trials_reads_the_optuna_shape():
    """Optuna records a state name rather than a completion flag."""
    from workbench.training.hpo_runner import summarize_trials

    counts = summarize_trials(
        [
            {"state": "COMPLETE", "value": 0.5},
            {"state": "PRUNED", "value": 0.8},
            {"state": "FAIL", "value": None},
        ]
    )
    assert counts == {"attempted": 3, "completed": 1, "pruned": 1, "failed": 1}


class _StubAdapter(HpoAdapter):
    """Objective = the candidate's `x` (default 5.0); optionally fails the baseline."""

    def __init__(self, fail_baseline=False):
        self.fail_baseline = fail_baseline

    def make_trial_fn(self, *, train_df, folds, val_df, hyperparameters, metric, concurrency):
        def trial_fn(config, report):
            if self.fail_baseline and "x" not in config:
                raise RuntimeError("baseline failed to train")
            value = float(config.get("x", hyperparameters.get("x", 5.0)))
            report(step=1, **{metric: value})
            return value

        return trial_fn


def _rerank(adapter, trials, **overrides):
    import pandas as pd

    from workbench.training.hpo_harness import FloatRange, HpoResult
    from workbench.training.hpo_runner import rerank_finalists

    result = HpoResult(
        best_config=trials[0]["config"],
        best_value=trials[0]["value"],
        metric="holdout_mae",
        mode="min",
        n_trials=len(trials),
        trials=trials,
    )
    kwargs = dict(
        top_k=5,
        adapter=adapter,
        train_df=pd.DataFrame({"y": [1.0, 2.0]}),
        val_df=pd.DataFrame({"y": [1.0]}),
        folds=[],
        split_kwargs={},
        base_hyperparameters={},
        metric="holdout_mae",
        search_space={"x": FloatRange(0.0, 10.0, default=5.0)},
        seed=42,
        backend="optuna",
        max_parallel=1,
        resources=None,
    )
    kwargs.update(overrides)
    return rerank_finalists(result, **kwargs)


_TRIALS = [
    {"number": 0, "value": 2.0, "state": "COMPLETE", "config": {"x": 2.0}},
    {"number": 1, "value": 4.0, "state": "COMPLETE", "config": {"x": 4.0}},
]


def test_rerank_picks_the_finalist_that_beats_the_baseline():
    best_config, info = _rerank(_StubAdapter(), _TRIALS)
    assert best_config == {"x": 2.0}
    assert info["baseline_value"] == 5.0 and info["best_value"] == 2.0


def test_rerank_tie_goes_to_the_baseline():
    trials = [{"number": 0, "value": 5.0, "state": "COMPLETE", "config": {"x": 5.0}}]
    best_config, _ = _rerank(_StubAdapter(), trials)
    assert best_config == {}


def test_rerank_publishes_the_baseline_when_it_fails_to_score():
    """No measured baseline means no finalist can prove itself — the caller's own
    hyperparameters ship, preserving the never-worse-than-untuned guarantee."""
    best_config, info = _rerank(_StubAdapter(fail_baseline=True), _TRIALS)
    assert best_config == {}
    assert info["baseline_value"] is None
    # The finalists' scores are still recorded for the audit trail.
    assert [r["holdout_mae"] for r in info["candidates"]] == [None, 2.0, 4.0]


def test_run_hpo_end_to_end_writes_the_artifact_contract(tmp_path):
    """A tiny real search (Optuna backend, stub objective) produces the full artifact set
    with the backend-independent schema: one `completed` bool column, a baseline row, and
    same-basis best/baseline values in best_config.json."""
    import json

    import pandas as pd

    from workbench.training.hpo_harness import FloatRange
    from workbench.training.hpo_runner import run_hpo

    train_df = pd.DataFrame({"feat": [float(i) for i in range(20)], "y": [float(i % 7) for i in range(20)]})
    published = run_hpo(
        train_df,
        train_df.iloc[0:0],
        {"n_folds": 2, "split_strategy": "random", "seed": 42, "hpo": {"n_trials": 4}},
        {"n_trials": 4, "backend": "optuna", "rerank_top_k": 2},
        adapter=_StubAdapter(),
        search_space={"x": FloatRange(0.0, 10.0, default=5.0)},
        primary_target="y",
        output_dir=str(tmp_path),
    )
    assert "hpo" not in published

    trials = pd.read_csv(tmp_path / "hpo_trials.csv")
    assert trials["completed"].dtype == bool
    assert "state" not in trials.columns
    assert (trials["kind"] == "baseline").sum() == 1
    assert len(trials) == 5  # 4 trials + the baseline row

    best = json.loads((tmp_path / "best_config.json").read_text())
    assert best["search_baseline_value"] == 5.0
    assert best["baseline_value"] == 5.0
    assert best["best_value"] is not None and best["best_value"] <= best["baseline_value"]
    assert (tmp_path / "hpo_rerank.csv").exists()
