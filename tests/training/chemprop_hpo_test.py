"""Fast unit tests for the pure pieces of ``workbench.training.chemprop_hpo``.

Covers the default search space and config merge — no chemprop/GPU needed (those
imports are deferred in the trial function, not at module top). The framework-agnostic
orchestration is covered by ``hpo_runner_test.py``.
"""

# Workbench Imports
from workbench.training.chemprop_hpo import (
    _dataloader_workers,
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
    assert space["depth"] == IntRange(2, 6, 1, default=6)  # chemprop {2,3,4,5,6}
    assert space["hidden_dim"] == IntRange(100, 2400, 100, default=700)  # chemprop floor of 300 extended to 100
    assert space["ffn_num_layers"] == IntRange(1, 3, 1, default=2)  # chemprop {1,2,3}
    # dropout is held out of the default space to keep the budget on the capacity knobs.
    assert "dropout" not in space
    assert isinstance(space["ffn_hidden_dim"], Choice)
    assert "1024-256-64" in space["ffn_hidden_dim"].options  # tapered head is a choice


def test_ffn_options_are_scalar_widths_or_parseable_shapes():
    """Every ffn_hidden_dim option is an int width or a dash-string shape (the pytorch
    `layers` convention) — one scalar cell per record, so the knob plots directly."""
    options = chemprop_search_space(("basic",))["ffn_hidden_dim"].options
    for opt in options:
        if isinstance(opt, str):
            widths = [int(x) for x in opt.split("-")]
            assert len(widths) >= 2 and all(w > 0 for w in widths)
        else:
            assert isinstance(opt, int) and opt > 0
    assert any(isinstance(o, str) for o in options)  # at least one tapered shape


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


def test_dataloader_workers_scale_down_with_concurrency():
    """Workers are shared vCPUs — more concurrent trials means fewer each, never zero."""
    assert _dataloader_workers(1) >= _dataloader_workers(8)
    assert _dataloader_workers(1000) == 1  # never zero, however tight the budget
    assert _dataloader_workers(0) >= 1  # guards a zero-concurrency call
    assert _dataloader_workers(1) <= 8  # capped regardless of core count


def test_spec_defaults_match_the_template():
    """Each knob's declared default must equal what the template would actually train with.

    The template's DEFAULT_HYPERPARAMETERS is what trains; a spec default is what the search
    records report for a knob nobody overrode. If the two disagree, the baseline row
    describes a model that was never built.
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

    mismatched = {
        knob: (spec.default, defaults.get(knob))
        for knob, spec in chemprop_search_space().items()
        if knob in defaults and spec.default != defaults[knob]
    }
    assert not mismatched, f"spec default != template default for {mismatched}"


def test_every_searched_knob_declares_a_default():
    """No knob may leave its default unset — that is what keeps search records NaN-free."""
    missing = [knob for knob, spec in chemprop_search_space().items() if spec.default is None]
    assert not missing, f"searched knobs with no declared default: {missing}"


def _adapter(targets):
    from workbench.training.chemprop_hpo import ChempropAdapter

    return ChempropAdapter(
        target_columns=targets,
        smiles_column="smiles",
        task="regression",
        model_type="uq_regressor",
        num_classes=None,
        task_weights=None,
    )


def test_single_task_packs_two_trials_per_gpu():
    assert _adapter(["pec50"]).resources_per_trial({}, "ray") == {"gpu": 0.5}


def test_multi_task_claims_a_whole_gpu():
    """Packing two multi-task trials on one card has been measured to OOM it."""
    assert _adapter(["pec50", "logd"]).resources_per_trial({}, "ray") == {"gpu": 1.0}


def test_explicit_gpus_per_trial_wins_over_the_default():
    block = {"gpus_per_trial": 0.25}
    assert _adapter(["pec50", "logd"]).resources_per_trial(block, "ray") == {"gpu": 0.25}


def test_optuna_backend_requests_no_ray_resources():
    assert _adapter(["pec50", "logd"]).resources_per_trial({}, "optuna") is None


def test_split_kwargs_names_the_molecule_column():
    """Scaffold/butina folds need the SMILES column; the adapter is what knows its name."""
    assert _adapter(["pec50"]).split_kwargs() == {"smiles_column": "smiles"}
