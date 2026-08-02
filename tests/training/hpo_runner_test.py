"""Fast unit tests for the framework-agnostic pieces of ``workbench.training.hpo_runner``.

Objective selection, shortlisting, the re-rank's publish decision, and the artifact
contract — no framework, no real training (a stub adapter stands in for one). The
per-framework objectives are covered by ``chemprop_hpo_test.py`` and ``xgb_hpo_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_runner import (
    HpoAdapter,
    best_config_record,
    resolve_max_parallel,
    shortlist_configs,
    trial_completed,
    trial_records,
    use_holdout,
)


def test_split_kwargs_default_is_empty():
    """The base adapter adds nothing to get_split_indices — frameworks with a molecule
    column (chemprop) override to name it."""
    assert HpoAdapter().split_kwargs() == {}


def test_max_parallel_divides_the_cards_by_the_per_trial_share():
    """The whole point of deriving it: concurrency is the box, not a hand-set number."""
    assert resolve_max_parallel({}, {"gpu": 0.5}, 4) == 8  # g6.12xlarge, single-task
    assert resolve_max_parallel({}, {"gpu": 1.0}, 4) == 4  # g6.12xlarge, multi-task


def test_max_parallel_stays_sane_on_a_one_gpu_box():
    """Laddering down to g6.2xlarge must pack the single card, not ask for cards it lacks."""
    assert resolve_max_parallel({}, {"gpu": 0.5}, 1) == 2
    assert resolve_max_parallel({}, {"gpu": 1.0}, 1) == 1


def test_max_parallel_falls_back_to_serial_without_gpus():
    """No GPU, or a backend that requested no resources, means one trial at a time."""
    assert resolve_max_parallel({}, {"gpu": 0.5}, 0) == 1  # CPU-only box
    assert resolve_max_parallel({}, None, 4) == 1  # optuna: no resource request
    assert resolve_max_parallel({}, {}, 4) == 1


def test_an_explicit_max_parallel_still_wins():
    """The derived value is a default, not a policy — a caller can override it."""
    assert resolve_max_parallel({"max_parallel": 3}, {"gpu": 0.5}, 4) == 3
    assert resolve_max_parallel({"max_parallel": 0}, {"gpu": 0.5}, 4) == 1  # never below 1


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


def _space():
    from workbench.training.hpo_harness import FloatRange, IntRange

    return {"x": FloatRange(0.0, 10.0, default=5.0), "depth": IntRange(2, 6, 1, default=6)}


def test_trial_records_normalize_both_backend_shapes():
    """Optuna reports a state name and Ray a flag; the CSV carries one `completed` bool
    either way, so a reader needs no per-backend logic."""
    optuna_rows = trial_records(
        [{"number": 0, "value": 0.4, "state": "COMPLETE", "config": {"x": 1.0}}], {}, _space(), 0.9
    )
    ray_rows = trial_records([{"number": 0, "value": 0.4, "completed": True, "config": {"x": 1.0}}], {}, _space(), 0.9)

    for rows in (optuna_rows, ray_rows):
        assert rows[0]["completed"] is True
        assert "state" not in rows[0]
    # Same schema from both backends
    assert set(optuna_rows[0]) == set(ray_rows[0])


def test_trial_records_mark_pruned_trials_incomplete():
    """A pruned trial keeps its (partial-ensemble) value but is not `completed`, which is
    what keeps it out of the shortlist and out of a reader's comparisons."""
    rows = trial_records(
        [
            {"number": 0, "value": 0.4, "state": "PRUNED", "config": {}},
            {"number": 1, "value": None, "state": "FAIL", "config": {}},
        ],
        {},
        _space(),
        0.9,
    )
    assert [r["completed"] for r in rows[:2]] == [False, False]
    assert rows[0]["value"] == 0.4 and rows[1]["value"] is None  # pruned vs failed stays recoverable


def test_trial_records_append_the_baseline_row():
    """The baseline is the caller's own hyperparameters on the search basis — the
    reference line a trials plot needs — and always completes."""
    rows = trial_records([{"number": 0, "value": 0.4, "state": "COMPLETE", "config": {"x": 1.0}}], {}, _space(), 0.9)
    baseline = rows[-1]
    assert baseline["kind"] == "baseline" and rows[0]["kind"] == "trial"
    assert baseline["number"] == -1
    assert baseline["value"] == 0.9
    assert baseline["completed"] is True


def test_trial_records_are_rectangular_and_nan_free():
    """Every row carries every searched knob at the value it trained at: the trial's own
    override, else the caller's hyperparameters, else the spec default."""
    rows = trial_records(
        [{"number": 0, "value": 0.4, "state": "COMPLETE", "config": {"x": 1.0}}],
        {"depth": 3},  # caller set depth; nobody set x on the baseline
        _space(),
        0.9,
    )
    assert rows[0]["hyperparameters"] == {"x": 1.0, "depth": 3}  # trial override wins for x
    assert rows[-1]["hyperparameters"] == {"x": 5.0, "depth": 3}  # baseline: spec default, caller's depth
    assert all(set(r["hyperparameters"]) == {"x", "depth"} for r in rows)


def test_best_config_record_keeps_the_two_bases_separate():
    """best/baseline come from the re-rank; search_* are phase 1's own numbers. Mixing
    them is the documented misread, so they must stay distinctly labelled."""
    from workbench.training.hpo_harness import HpoResult

    result = HpoResult(best_config={"x": 1.0}, best_value=0.30, metric="cv_mae", mode="min", n_trials=4, trials=[])
    counts = {"attempted": 4, "completed": 3, "pruned": 1, "failed": 0}
    rerank = {
        "best_value": 0.42,
        "baseline_value": 0.45,
        "fresh_split": True,
        "candidates": [{"candidate": "baseline"}],
    }
    record = best_config_record(
        result,
        metric="cv_mae",
        counts=counts,
        best_config={"x": 1.0},
        rerank=rerank,
        search_baseline_value=0.50,
    )

    # Re-rank basis: the margin the publish decision turned on
    assert (record["best_value"], record["baseline_value"]) == (0.42, 0.45)
    # Search basis: not comparable to the above when the partition was re-drawn
    assert (record["search_best_value"], record["search_baseline_value"]) == (0.30, 0.50)
    assert record["rerank_fresh_split"] is True
    assert record["trial_counts"] == counts
    assert record["rerank"] == rerank["candidates"]


def test_best_config_record_when_the_rerank_was_skipped():
    """A disabled/failed re-rank leaves an empty info dict — the record still writes, with
    the re-rank basis absent rather than silently borrowing the search's numbers."""
    from workbench.training.hpo_harness import HpoResult

    result = HpoResult(best_config={}, best_value=0.30, metric="cv_mae", mode="min", n_trials=1, trials=[])
    record = best_config_record(
        result,
        metric="cv_mae",
        counts={"attempted": 1, "completed": 1, "pruned": 0, "failed": 0},
        best_config={},
        rerank={},
        search_baseline_value=0.50,
    )
    assert record["best_value"] is None and record["baseline_value"] is None
    assert record["rerank"] == [] and record["rerank_fresh_split"] is False


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
