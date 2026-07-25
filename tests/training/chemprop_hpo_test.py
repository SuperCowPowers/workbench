"""Fast unit tests for the pure pieces of ``workbench.training.chemprop_hpo``.

Covers the default search space and config merge — no chemprop/GPU needed (those
imports are deferred in the search entry point, not at module top).
"""

# Workbench Imports
from workbench.training.chemprop_hpo import (
    _dataloader_workers,
    _shortlist_configs,
    _use_holdout,
    _trial_completed,
    chemprop_search_space,
    merge_best_config,
    resolve_search_space,
)
from workbench.training.hpo_harness import Choice, IntRange


def test_default_space_is_basic_plus_lr():
    """The default space is both groups: capacity + LR schedule + batch size."""
    space = chemprop_search_space()
    assert set(space) == {
        "depth",
        "hidden_dim",
        "ffn_num_layers",
        "ffn_hidden_dim",
        "max_lr",
        "warmup_epochs",
        "batch_size",
    }
    assert space["depth"] == IntRange(2, 6, 1)  # chemprop {2,3,4,5,6}
    assert space["hidden_dim"] == IntRange(100, 2400, 100)  # chemprop floor of 300 extended to 100
    assert space["ffn_num_layers"] == IntRange(1, 3, 1)  # chemprop {1,2,3}
    # dropout is held out of the default space to keep the budget on the capacity knobs.
    assert "dropout" not in space
    assert isinstance(space["ffn_hidden_dim"], Choice)
    assert [1024, 256, 64] in space["ffn_hidden_dim"].options  # tapered head is a choice


def test_lr_group_adds_schedule_knobs():
    """The lr group carries max_lr (log), warmup_epochs, and batch_size."""
    space = chemprop_search_space(("lr",))
    assert set(space) == {"max_lr", "warmup_epochs", "batch_size"}
    assert space["max_lr"].log is True
    assert isinstance(space["batch_size"], Choice)


def test_unknown_group_raises():
    try:
        chemprop_search_space(("bogus",))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_search_space_shorthands():
    """String/iterable/dict/None all resolve to a {knob: Spec} space."""
    assert set(resolve_search_space("basic")) == set(chemprop_search_space(("basic",)))
    assert "max_lr" in resolve_search_space("basic+lr")
    # basic+lr is a superset of basic
    assert set(resolve_search_space("basic")) < set(resolve_search_space("basic+lr"))
    assert set(resolve_search_space(None)) == set(chemprop_search_space())
    custom = {"depth": IntRange(3, 5, 1)}
    assert resolve_search_space(custom) is custom  # ready dict passes through


def test_merge_drops_hpo_block_and_applies_winner():
    """Merge overlays the winner and strips the hpo block."""
    hp = {"uq_version": "v1", "depth": 6, "hpo": {"n_trials": 40}}
    best = {"depth": 3, "dropout": 0.15}
    merged = merge_best_config(hp, best)
    assert "hpo" not in merged
    assert merged["depth"] == 3  # winner overrides base
    assert merged["dropout"] == 0.15
    assert merged["uq_version"] == "v1"  # untouched base knob preserved


def test_merge_ties_lr_schedule_to_max_lr():
    """When max_lr is searched, init_lr/final_lr are tied to it (one-tenth)."""
    merged = merge_best_config({"uq_version": "v1"}, {"max_lr": 2e-3})
    assert merged["init_lr"] == 2e-4
    assert merged["final_lr"] == 2e-4


def test_merge_leaves_lr_alone_when_not_searched():
    """No max_lr in the winner → no derived init_lr/final_lr injected."""
    merged = merge_best_config({"uq_version": "v1", "init_lr": 1e-4}, {"depth": 4})
    assert merged["init_lr"] == 1e-4  # base value preserved, not overwritten
    assert "final_lr" not in merged


def test_trial_completed_reads_both_backend_shapes():
    """Ray records a `completed` flag; Optuna records a state name."""
    assert _trial_completed({"completed": True}) is True
    assert _trial_completed({"completed": False}) is False
    assert _trial_completed({"state": "COMPLETE"}) is True
    assert _trial_completed({"state": "PRUNED"}) is False
    assert _trial_completed({}) is False  # neither shape → not completed


def test_shortlist_ranks_completed_trials_best_first():
    """Pruned/unscored trials are excluded and the rest sort by objective."""
    trials = [
        {"value": 0.9, "state": "COMPLETE", "config": {"depth": 2}},
        {"value": 0.5, "state": "COMPLETE", "config": {"depth": 3}},
        {"value": 0.1, "state": "PRUNED", "config": {"depth": 4}},  # pruned value is partial
        {"value": None, "state": "COMPLETE", "config": {"depth": 5}},
        {"value": 0.7, "state": "COMPLETE", "config": {"depth": 6}},
    ]
    assert _shortlist_configs(trials, 3) == [{"depth": 3}, {"depth": 6}, {"depth": 2}]
    assert _shortlist_configs(trials, 1) == [{"depth": 3}]


def test_shortlist_dedupes_and_handles_unhashable_configs():
    """A repeated config is listed once, including configs holding tapered ffn lists."""
    tapered = {"ffn_hidden_dim": [1024, 256, 64]}
    trials = [
        {"value": 0.4, "completed": True, "config": tapered},
        {"value": 0.6, "completed": True, "config": dict(tapered)},  # same config, worse draw
        {"value": 0.5, "completed": True, "config": {"ffn_hidden_dim": 600}},
    ]
    assert _shortlist_configs(trials, 5) == [tapered, {"ffn_hidden_dim": 600}]


def test_shortlist_empty_when_everything_pruned():
    """Nothing completed → no finalists (caller falls back to the search winner)."""
    assert _shortlist_configs([{"value": 0.3, "completed": False, "config": {"depth": 2}}], 5) == []


def test_use_holdout_defaults_to_out_of_fold():
    """Default never tunes on the validation rows — a benchmark holdout stays uncontaminated."""
    assert _use_holdout(None, 500) is False
    assert _use_holdout(None, 0) is False


def test_cv_mae_override_ignores_the_holdout():
    """Asking for cv_mae explicitly matches the default."""
    assert _use_holdout("cv_mae", 500) is False
    assert _use_holdout("cv_mae", 0) is False


def test_holdout_mae_is_opt_in_and_requires_validation_rows():
    """Tuning toward the holdout happens only when asked for, and only if rows exist."""
    assert _use_holdout("holdout_mae", 500) is True
    try:
        _use_holdout("holdout_mae", 0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "validation_ids" in str(exc)


def test_unknown_metric_raises():
    """A typo'd metric fails rather than silently falling back to a default objective."""
    try:
        _use_holdout("holdout_rmse", 500)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must be" in str(exc)


def test_dataloader_workers_scale_down_with_concurrency():
    """Workers are shared vCPUs — more concurrent trials means fewer each, never zero."""
    assert _dataloader_workers(1) >= _dataloader_workers(8)
    assert _dataloader_workers(1000) == 1  # never zero, however tight the budget
    assert _dataloader_workers(0) >= 1  # guards a zero-concurrency call
    assert _dataloader_workers(1) <= 8  # capped regardless of core count


def test_every_searched_knob_has_a_template_default():
    """Each knob in the default space must have a value in the template's defaults.

    The re-rank records the effective value of every searched knob, falling back to the base
    hyperparameters for knobs a candidate didn't override. A knob with no template default
    resolves to None there and lands as NaN in hpo_rerank.csv, which downstream readers
    would have to special-case. Adding a knob to _SEARCH_GROUPS means giving it a default.
    """
    import ast
    from pathlib import Path

    template = Path(__file__).parents[2] / "src/workbench/model_scripts/chemprop/chemprop.template"
    tree = ast.parse(template.read_text())
    defaults = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "DEFAULT_HYPERPARAMETERS" for t in node.targets)
    )

    missing = [knob for knob in chemprop_search_space() if knob not in defaults]
    assert not missing, f"searched knobs with no template default (would render as NaN): {missing}"
