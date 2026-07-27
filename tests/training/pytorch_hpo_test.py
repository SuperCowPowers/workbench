"""Fast unit tests for the pure pieces of ``workbench.training.pytorch_hpo``.

Covers the default search space — no torch training. The framework-agnostic orchestration
is covered by ``hpo_runner_test.py``.
"""

# Workbench Imports
from workbench.training.hpo_harness import Choice, FloatRange
from workbench.training.pytorch_hpo import pytorch_search_space, resolve_search_space


def test_default_space_is_basic_plus_optimizer():
    """The default space is both groups: architecture + optimizer."""
    space = pytorch_search_space()
    assert set(space) == {"layers", "dropout", "learning_rate", "weight_decay", "batch_size"}
    assert isinstance(space["layers"], Choice)
    assert space["learning_rate"].log is True
    assert space["weight_decay"].log is True
    assert space["dropout"] == FloatRange(0.0, 0.4, step=0.05, default=0.05)


def test_layer_shapes_are_parseable_and_tapered_options_exist():
    """Every shape must split into ints the way the template parses `layers`."""
    options = pytorch_search_space(("basic",))["layers"].options
    for shape in options:
        widths = [int(x) for x in shape.split("-")]
        assert widths and all(w > 0 for w in widths)
    assert any(len(s.split("-")) == 4 for s in options)  # a deep option
    assert any(len(s.split("-")) == 2 for s in options)  # a shallow option


def test_epoch_and_ensemble_knobs_are_excluded():
    """Early stopping owns the epoch budget, and restore_best_weights is a UQ decision."""
    space = pytorch_search_space()
    for excluded in ("max_epochs", "early_stopping_patience", "restore_best_weights", "loss"):
        assert excluded not in space


def test_unknown_group_raises():
    try:
        pytorch_search_space(("bogus",))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolve_search_space_shorthands():
    """String/iterable/dict/None all resolve to a {knob: Spec} space."""
    assert set(resolve_search_space("basic")) == set(pytorch_search_space(("basic",)))
    assert "learning_rate" in resolve_search_space("basic+optimizer")
    assert set(resolve_search_space("basic")) < set(resolve_search_space("basic+optimizer"))
    assert set(resolve_search_space(None)) == set(pytorch_search_space())
    custom = {"dropout": FloatRange(0.0, 0.2)}
    assert resolve_search_space(custom) is custom  # ready dict passes through


def test_spec_defaults_match_the_template():
    """Each knob's declared default must equal what the template would actually train with."""
    import ast
    from pathlib import Path

    template = Path(__file__).parents[2] / "src/workbench/model_scripts/pytorch_model/pytorch.template"
    tree = ast.parse(template.read_text())
    defaults = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "DEFAULT_HYPERPARAMETERS" for t in node.targets)
    )

    mismatched = {
        knob: (spec.default, defaults.get(knob))
        for knob, spec in pytorch_search_space().items()
        if knob in defaults and spec.default != defaults[knob]
    }
    assert not mismatched, f"spec default != template default for {mismatched}"


def test_every_searched_knob_declares_a_default():
    """No knob may leave its default unset — that is what keeps search records NaN-free."""
    missing = [knob for knob, spec in pytorch_search_space().items() if spec.default is None]
    assert not missing, f"searched knobs with no declared default: {missing}"


def test_default_layer_shape_is_in_the_options():
    """The baseline must be reachable by the search, or trials can't reproduce it."""
    space = pytorch_search_space()
    assert space["layers"].default in space["layers"].options
