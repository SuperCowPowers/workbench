"""Fast unit tests for the framework-agnostic pieces of ``workbench.training.hpo_runner``.

Trial-record normalization, baseline identification, and the artifact contract — no
framework, no real training (a stub adapter stands in for one). The per-framework
objectives are covered by ``chemprop_hpo_test.py`` and ``xgb_hpo_test.py``.
"""

# Workbench Imports
import pytest

from workbench.training.hpo_runner import (
    METRIC,
    check_hpo_block,
    HpoAdapter,
    baseline_value,
    best_config_record,
    resolve_max_parallel,
    summarize_trials,
    trial_completed,
    trial_records,
)


def test_base_adapter_defaults():
    """Frameworks override what they need; the rest has to work untouched."""
    adapter = HpoAdapter()
    assert adapter.split_kwargs() == {}  # chemprop overrides to name its SMILES column
    assert adapter.resources_per_trial({}, "ray") is None
    # merge overlays the winner and strips the hpo block
    assert adapter.merge_config({"max_depth": 7, "hpo": {"n_trials": 40}}, {"max_depth": 3}) == {"max_depth": 3}


def test_max_parallel_divides_the_cards_by_the_per_trial_share():
    """The whole point of deriving it: concurrency is the box, not a hand-set number."""
    assert resolve_max_parallel({}, {"gpu": 0.5}, 4) == 8  # g6.12xlarge, single-task
    assert resolve_max_parallel({}, {"gpu": 1.0}, 4) == 4  # g6.12xlarge, multi-task
    assert resolve_max_parallel({}, {"gpu": 0.5}, 1) == 2  # one card, packed


def test_max_parallel_falls_back_to_serial_without_a_gpu_share():
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
    assert trial_completed({"state": "FAIL"}) is False
    assert trial_completed({}) is False  # neither shape → not completed


def test_summarize_trials_separates_stopped_from_died():
    """Three outcomes, not two: only `completed` is rankable, and a stopped trial still
    carries the partial value that a failed one never produced."""
    ray_shape = [
        {"completed": True, "value": 0.5},
        {"completed": False, "value": 0.9},  # stopped at a rung
        {"completed": False, "value": None},  # raised
    ]
    assert summarize_trials(ray_shape) == {"attempted": 3, "completed": 1, "pruned": 1, "failed": 1}
    optuna_shape = [
        {"state": "COMPLETE", "value": 0.5},
        {"state": "PRUNED", "value": 0.9},
        {"state": "FAIL", "value": None},
    ]
    assert summarize_trials(optuna_shape) == {"attempted": 3, "completed": 1, "pruned": 1, "failed": 1}


def _space():
    from workbench.training.hpo_harness import FloatRange, IntRange

    return {"x": FloatRange(0.0, 10.0, default=5.0), "depth": IntRange(2, 6, 1, default=6)}


def test_trial_records_normalize_both_backend_shapes():
    """One `completed` bool column either way, so a reader needs no per-backend logic."""
    optuna_rows = trial_records([{"number": 0, "value": 0.4, "state": "COMPLETE", "config": {"x": 1.0}}], {}, _space())
    ray_rows = trial_records([{"number": 0, "value": 0.4, "completed": True, "config": {"x": 1.0}}], {}, _space())

    for rows in (optuna_rows, ray_rows):
        assert rows[0]["completed"] is True
        assert "state" not in rows[0]
    assert set(optuna_rows[0]) == set(ray_rows[0])


def test_trial_records_keep_a_dead_trials_row_without_a_value():
    """A trial that never scored still belongs in the record, marked incomplete."""
    rows = trial_records([{"number": 0, "value": None, "state": "FAIL", "config": {}}], {}, _space())
    assert rows[0]["completed"] is False and rows[0]["value"] is None


def test_trial_records_are_rectangular_and_mark_the_baseline():
    """Every row carries every searched knob at the value it trained at — the trial's own
    override, else the caller's, else the spec default — and the trial that lands on the
    caller's own settings is the baseline the plots center on."""
    trials = [
        {"number": 0, "value": 0.9, "state": "COMPLETE", "config": {"x": 5.0}},  # the seeded baseline
        {"number": 1, "value": 0.4, "state": "COMPLETE", "config": {"x": 1.0}},
    ]
    rows = trial_records(trials, {"depth": 3}, _space())  # caller set depth; nobody set x

    assert [r["kind"] for r in rows] == ["baseline", "trial"]
    assert rows[0]["hyperparameters"] == {"x": 5.0, "depth": 3}  # spec default, caller's depth
    assert rows[1]["hyperparameters"] == {"x": 1.0, "depth": 3}  # trial override wins for x
    assert all(set(r["hyperparameters"]) == {"x", "depth"} for r in rows)
    assert baseline_value(rows) == 0.9


def test_only_the_first_matching_row_is_the_baseline():
    """A sampler can land on the caller's own config again on a discrete space. A second
    `baseline` row would drop a real trial out of the plots (which filter on kind) and make
    the reference line ambiguous."""
    same = {"number": 0, "value": 0.9, "state": "COMPLETE", "config": {"x": 5.0}}
    rows = trial_records(
        [same, dict(same, number=1, value=0.8), {**same, "number": 2, "config": {"x": 1.0}}], {}, _space()
    )
    assert [r["kind"] for r in rows] == ["baseline", "trial", "trial"]
    assert baseline_value(rows) == 0.9  # the seeded one, which ran first


def test_baseline_value_is_none_when_the_baseline_never_scored():
    """A failed baseline costs the plots their reference line, not the search its winner."""
    rows = trial_records([{"number": 0, "value": None, "state": "FAIL", "config": {"x": 5.0}}], {}, _space())
    assert rows[0]["kind"] == "baseline"
    assert baseline_value(rows) is None


def test_best_config_record_labels_the_search_basis():
    """The `search_` prefixes are the contract: these are the search's own numbers on the
    folds it selected against, not an estimate of what the published model is worth."""
    from workbench.training.hpo_harness import HpoResult

    result = HpoResult(best_config={"x": 1.0}, best_value=0.30, metric=METRIC, mode="min", n_trials=4, trials=[])
    counts = {"attempted": 4, "completed": 4, "failed": 0}
    record = best_config_record(result, counts=counts, baseline=0.50)

    assert record["best_config"] == {"x": 1.0}
    assert (record["search_best_value"], record["search_baseline_value"]) == (0.30, 0.50)
    assert record["metric"] == METRIC
    assert record["trial_counts"] == counts


class _StubAdapter(HpoAdapter):
    """Objective = the candidate's `x` (default 5.0)."""

    def make_trial_fn(self, *, train_df, folds, hyperparameters, concurrency):
        def trial_fn(config, report):
            value = float(config.get("x", hyperparameters.get("x", 5.0)))
            # Mimic an ensemble firming up: the running value converges on the final one.
            for fold in range(1, len(folds) + 1):
                report(fold, value + (len(folds) - fold) * 0.01)
            return value

        return trial_fn


def test_run_hpo_end_to_end_writes_the_artifact_contract(tmp_path):
    """A tiny real search (Optuna backend, stub objective) produces the artifact set with
    the backend-independent schema: one `completed` bool column, exactly one baseline row
    carrying the caller's own settings, and the search's basis in best_config.json."""
    import json

    import pandas as pd

    from workbench.training.hpo_harness import FloatRange
    from workbench.training.hpo_runner import run_hpo

    train_df = pd.DataFrame({"feat": [float(i) for i in range(20)], "y": [float(i % 7) for i in range(20)]})
    published = run_hpo(
        train_df,
        {"n_folds": 2, "split_strategy": "random", "seed": 42, "hpo": {"n_trials": 6}},
        {"n_trials": 6, "backend": "optuna"},
        adapter=_StubAdapter(),
        search_space={"x": FloatRange(0.0, 10.0, default=5.0)},
        primary_target="y",
        output_dir=str(tmp_path),
    )
    assert "hpo" not in published

    trials = pd.read_csv(tmp_path / "hpo_trials.csv")
    assert trials["completed"].dtype == bool
    assert "state" not in trials.columns
    assert len(trials) == 6  # the baseline is one of them, not an extra row
    assert (trials["kind"] == "baseline").sum() == 1

    best = json.loads((tmp_path / "best_config.json").read_text())
    assert best["metric"] == METRIC
    assert best["search_baseline_value"] == 5.0  # the caller's own hyperparameters
    assert best["search_best_value"] <= best["search_baseline_value"]
    assert not (tmp_path / "hpo_rerank.csv").exists()


def test_the_trajectory_records_every_rung_a_trial_reported_at(tmp_path):
    """The history ends where the trial did and agrees with its `value` there. Every trial
    has one, the baseline included: its exemption is that the pruner is never consulted
    about it, not that it stays quiet."""
    import json

    import pandas as pd

    from workbench.training.hpo_harness import FloatRange
    from workbench.training.hpo_runner import run_hpo

    train_df = pd.DataFrame({"feat": [float(i) for i in range(20)], "y": [float(i % 7) for i in range(20)]})
    run_hpo(
        train_df,
        {"n_folds": 2, "split_strategy": "random", "seed": 42, "hpo": {"n_trials": 8}},
        {"n_trials": 8, "backend": "optuna"},
        adapter=_StubAdapter(),
        search_space={"x": FloatRange(0.0, 10.0, default=5.0)},
        primary_target="y",
        output_dir=str(tmp_path),
    )
    trials = pd.read_csv(tmp_path / "hpo_trials.csv")
    trials["trajectory"] = [{int(k): v for k, v in json.loads(cell).items()} for cell in trials["trajectory"]]

    for _, row in trials.iterrows():
        assert max(row["trajectory"]) == row["step"]  # the history ends where the trial did
        assert row["trajectory"][int(row["step"])] == pytest.approx(row["value"])
        # The stub converges on its final value, so an earlier fold reads higher.
        assert all(row["trajectory"][s] > row["value"] for s in row["trajectory"] if s < row["step"])

    assert (trials["step"] < 2).any(), "no trial was stopped at a rung — the ladder never engaged"
    assert (trials[trials["completed"]]["trajectory"].map(len) == 2).all()
    assert list(trials[trials["kind"].eq("baseline")]["trajectory"].iloc[0]) == [1, 2]


def test_a_retired_hpo_key_fails_loudly():
    """Silently ignoring it would run a different objective, or a different compute bill,
    on a job that costs hours."""
    for key in ("metric", "rerank_top_k", "n_folds"):
        with pytest.raises(ValueError, match="no longer supported"):
            check_hpo_block({key: "whatever"})


def test_a_misspelled_hpo_key_fails_loudly():
    """`n_trails` would otherwise sail through and run the default budget."""
    with pytest.raises(ValueError, match="unknown hpo key"):
        check_hpo_block({"n_trails": 100})


def test_the_supported_hpo_keys_pass():
    check_hpo_block({"n_trials": 60, "backend": "ray", "search_space": {}, "max_parallel": 4, "gpus_per_trial": 0.5})
    check_hpo_block({})  # an empty block is a real request: search on every default
