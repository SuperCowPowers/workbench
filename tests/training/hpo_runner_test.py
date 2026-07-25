"""Fast unit tests for the framework-agnostic pieces of ``workbench.training.hpo_runner``.

Objective selection, shortlisting, and the default config merge — no framework, no
training. The per-framework objectives are covered by ``chemprop_hpo_test.py`` and
``xgb_hpo_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_runner import HpoAdapter, shortlist_configs, trial_completed, use_holdout


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
