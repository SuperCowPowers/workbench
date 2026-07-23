"""Fast unit tests for the pure pieces of ``workbench.training.chemprop_hpo``.

Covers the default search space and config merge — no chemprop/GPU needed (those
imports are deferred in the search entry point, not at module top).
"""

# Workbench Imports
from workbench.training.chemprop_hpo import chemprop_search_space, merge_best_config, resolve_search_space
from workbench.training.hpo_harness import Choice, FloatRange, IntRange


def test_basic_group_is_default():
    """The default space is the `basic` group with our chosen knobs/ranges."""
    space = chemprop_search_space()
    assert set(space) == {"depth", "hidden_dim", "dropout", "ffn_hidden_dim"}
    assert space["depth"] == IntRange(2, 6, 1)
    assert space["dropout"] == FloatRange(0.0, 0.3, 0.05)
    assert isinstance(space["ffn_hidden_dim"], Choice)
    assert [1024, 256, 64] in space["ffn_hidden_dim"].options  # tapered head is a choice


def test_lr_group_adds_schedule_knobs():
    """basic+lr adds max_lr (log) and warmup_epochs on top of basic."""
    space = chemprop_search_space(("basic", "lr"))
    assert "max_lr" in space and "warmup_epochs" in space
    assert space["max_lr"].log is True


def test_unknown_group_raises():
    try:
        chemprop_search_space(("bogus",))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_search_space_shorthands():
    """String/iterable/dict/None all resolve to a {knob: Spec} space."""
    assert set(resolve_search_space("basic")) == {"depth", "hidden_dim", "dropout", "ffn_hidden_dim"}
    assert "max_lr" in resolve_search_space("basic+lr")
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
