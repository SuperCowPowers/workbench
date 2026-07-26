"""Fast unit tests for the pure pieces of ``workbench.training.xgb_hpo``.

Covers the default search space and the estimator-kwarg reduction — no XGBoost training.
The framework-agnostic orchestration is covered by ``hpo_runner_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_harness import FloatRange, IntRange
from workbench.training.xgb_hpo import _xgb_threads, resolve_search_space, xgb_params, xgb_search_space


def test_default_space_is_basic_plus_reg():
    """The default space is both groups: capacity/boosting + sampling/regularization."""
    space = xgb_search_space()
    assert set(space) == {
        "max_depth",
        "min_child_weight",
        "n_estimators",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }
    assert space["max_depth"] == IntRange(3, 12, 1, default=7)
    assert space["learning_rate"].log is True  # rate spans an order of magnitude
    assert space["gamma"].log is False  # starts at 0, so it cannot be sampled log-uniformly
    assert space["reg_alpha"].log is True


def test_reg_group_is_sampling_and_penalties():
    space = xgb_search_space(("reg",))
    assert set(space) == {"subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"}
    assert space["subsample"] == FloatRange(0.5, 1.0, step=0.05, default=0.8)


def test_unknown_group_raises():
    try:
        xgb_search_space(("bogus",))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_search_space_shorthands():
    """String/iterable/dict/None all resolve to a {knob: Spec} space."""
    assert set(resolve_search_space("basic")) == set(xgb_search_space(("basic",)))
    assert "subsample" in resolve_search_space("basic+reg")
    assert set(resolve_search_space("basic")) < set(resolve_search_space("basic+reg"))
    assert set(resolve_search_space(None)) == set(xgb_search_space())
    custom = {"max_depth": IntRange(3, 5, 1)}
    assert resolve_search_space(custom) is custom  # ready dict passes through


def test_xgb_params_drops_workbench_knobs_and_the_hpo_block():
    """Only estimator kwargs reach XGBoost — the template's own knobs are filtered out."""
    params = xgb_params(
        {"n_folds": 5, "split_strategy": "scaffold", "uq_version": "v1", "max_depth": 4, "hpo": {"n_trials": 10}}
    )
    assert params["max_depth"] == 4
    for dropped in ("n_folds", "split_strategy", "uq_version", "hpo"):
        assert dropped not in params


def test_xgb_params_maps_seed_and_offsets_per_fold():
    """`seed` becomes random_state, offset per fold so ensemble members differ."""
    assert xgb_params({"seed": 42})["random_state"] == 42
    assert xgb_params({"seed": 42}, fold_idx=3)["random_state"] == 45
    assert "seed" not in xgb_params({"seed": 42})
    assert xgb_params({})["random_state"] == 42  # template default when unset


def test_xgb_params_defaults_to_mae_objective():
    """Regression defaults to MAE, matching the template, but never overrides a caller."""
    assert xgb_params({})["objective"] == "reg:absoluteerror"
    assert xgb_params({"objective": "reg:squarederror"})["objective"] == "reg:squarederror"


def test_threads_scale_down_with_concurrency():
    """Cores are shared — more concurrent trials means fewer threads each, never zero."""
    assert _xgb_threads(1) >= _xgb_threads(8)
    assert _xgb_threads(1000) == 1  # never zero, however tight the budget
    assert _xgb_threads(0) >= 1  # guards a zero-concurrency call


def test_spec_defaults_match_the_template():
    """Each knob's declared default must equal what the template would actually train with.

    The template's DEFAULT_HYPERPARAMETERS is what trains; a spec default is what the search
    records report for a knob nobody overrode. If the two disagree, the baseline row
    describes a model that was never built.
    """
    import ast
    from pathlib import Path

    template = Path(__file__).parents[2] / "src/workbench/model_scripts/xgb_model/xgb_model.template"
    tree = ast.parse(template.read_text())
    defaults = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "DEFAULT_HYPERPARAMETERS" for t in node.targets)
    )

    mismatched = {
        knob: (spec.default, defaults.get(knob))
        for knob, spec in xgb_search_space().items()
        if knob in defaults and spec.default != defaults[knob]
    }
    assert not mismatched, f"spec default != template default for {mismatched}"


def test_every_searched_knob_declares_a_default():
    """No knob may leave its default unset — that is what keeps search records NaN-free."""
    missing = [knob for knob, spec in xgb_search_space().items() if spec.default is None]
    assert not missing, f"searched knobs with no declared default: {missing}"
