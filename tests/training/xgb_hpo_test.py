"""Fast unit tests for the pure pieces of ``workbench.training.xgb_hpo``.

Covers the default search space and the estimator-kwarg reduction — no XGBoost training.
The framework-agnostic orchestration is covered by ``hpo_runner_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_harness import FloatRange, IntRange
from workbench.training.xgb_core import align_frame, xgb_params
from workbench.training.xgb_hpo import XGBAdapter, _xgb_threads, resolve_search_space, xgb_search_space


def test_default_space_is_basic_plus_reg():
    """The default space is both groups: capacity/boosting + sampling/regularization."""
    space = xgb_search_space()
    assert set(space) == {
        "max_depth",
        "min_child_weight",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }
    # Early stopping owns the tree budget, so searching it would only fight that.
    assert "n_estimators" not in space
    assert space["max_depth"] == IntRange(3, 16, 1, default=7)
    assert space["learning_rate"].log is True  # rate spans an order of magnitude
    assert space["gamma"].log is False  # starts at 0, so it cannot be sampled log-uniformly
    assert space["reg_alpha"].log is True


def test_reg_group_is_sampling_and_penalties():
    space = xgb_search_space(("reg",))
    assert set(space) == {"subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"}
    assert space["subsample"] == FloatRange(0.4, 1.0, step=0.05, default=0.8)


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
        {
            "n_folds": 5,
            "split_strategy": "scaffold",
            "uq_version": "v1",
            "early_stopping_fraction": 0.1,
            "max_depth": 4,
            "hpo": {"n_trials": 10},
        }
    )
    assert params["max_depth"] == 4
    for dropped in ("n_folds", "split_strategy", "uq_version", "early_stopping_fraction", "hpo"):
        assert dropped not in params
    # early_stopping_rounds IS an estimator kwarg, so it must survive the reduction.
    assert xgb_params({"early_stopping_rounds": 50})["early_stopping_rounds"] == 50


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


def test_align_frame_applies_fitted_transforms_and_is_idempotent():
    """A raw holdout gets the training mappings + decompression; a second pass is a no-op."""
    import pandas as pd

    df = pd.DataFrame({"x1": [1.0, 2.0], "color": ["blue", "red"], "fingerprint": ["10", "01"], "y": [0.1, 0.2]})
    once = align_frame(df, {"color": ["red", "blue"]}, ["x1", "color", "fingerprint"], ["fingerprint"])

    assert str(once["color"].dtype) == "category"
    assert "fingerprint" not in once.columns
    assert {"fin_0", "fin_1"} <= set(once.columns)
    assert "fingerprint" in df.columns  # input not mutated

    twice = align_frame(once, {"color": ["red", "blue"]}, ["x1", "color", "fingerprint"], ["fingerprint"])
    pd.testing.assert_frame_equal(once, twice)


def test_align_frame_handles_the_empty_holdout():
    """No validation_ids means an empty holdout that still has the compressed column —
    it must pass through rather than trip decompression's 0-row guard."""
    import pandas as pd

    empty = pd.DataFrame({"x1": pd.Series(dtype=float), "fingerprint": pd.Series(dtype=object)})
    out = align_frame(empty, {"color": ["red"]}, ["x1", "fingerprint"], ["fingerprint"])
    assert len(out) == 0 and "fingerprint" in out.columns


def test_adapter_prepare_frame_without_alignment_state_passes_through():
    """The minimal adapter (unit-test scale) must not require the template's fitted state."""
    import pandas as pd

    adapter = XGBAdapter(target="y", features=["x1"])
    df = pd.DataFrame({"x1": [1.0, 2.0], "y": [0.1, 0.2]})
    pd.testing.assert_frame_equal(adapter.prepare_frame(df), df)
