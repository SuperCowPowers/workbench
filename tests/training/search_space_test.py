"""Tests for SearchSpace — the JSON wire form users pass to hpo["search_space"].

No AWS and no training deps: the framework space modules defer their optuna/framework
imports, which is what lets a lean environment describe and edit a space.
"""

import json

import pytest

# Workbench Imports
from workbench.training.hpo_harness import Choice, FloatRange, IntRange, SearchSpace, spec_from_dict

FRAMEWORKS = ["chemprop", "xgboost", "pytorch"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_wire_form_round_trips(framework):
    """A space survives to_dict -> JSON -> from_dict unchanged. This is the whole contract:
    the dict is str()'d into the generated training script and parsed back there."""
    space = SearchSpace(framework)
    wire = space.to_dict()

    assert json.loads(json.dumps(wire)) == wire  # genuinely JSON, not a Python literal
    assert SearchSpace.from_dict(wire) == space


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_resolve_search_space_accepts_the_wire_form(framework):
    """The framework's own entry point — what the runner calls — takes the JSON."""
    import importlib

    from workbench.training.hpo_harness import SEARCH_SPACE_MODULES

    module = importlib.import_module(SEARCH_SPACE_MODULES[framework])
    space = SearchSpace(framework)

    assert module.resolve_search_space(space.to_dict()) == space
    assert module.resolve_search_space(dict(space)) == space  # a {knob: Spec} dict still passes
    assert module.resolve_search_space(None) == space  # and None is still the default


def test_editing_a_space_is_ordinary_dict_work():
    """The point of the class: start from ours, change your mind about one knob."""
    space = SearchSpace("chemprop")
    original = len(space)

    space["max_lr"] = FloatRange(1e-4, 1e-2, log=True, default=3e-3)
    del space["ffn_num_layers"]

    assert len(space) == original - 1
    assert space.to_dict()["max_lr"] == {"dist": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 3e-3}
    assert SearchSpace.from_dict(space.to_dict()) == space


def test_a_plain_dict_is_still_a_search_space():
    """Subclassing dict is what keeps every existing {knob: Spec} literal working."""
    space = SearchSpace(knobs={"depth": IntRange(2, 6, default=4)})
    assert isinstance(space, dict)
    assert space == {"depth": IntRange(2, 6, default=4)}


def test_framework_and_knobs_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        SearchSpace("chemprop", knobs={"depth": IntRange(2, 6)})


def test_unknown_framework_names_the_known_ones():
    with pytest.raises(ValueError, match="chemprop"):
        SearchSpace("catboost")


def test_subset_narrows_to_named_groups():
    full = SearchSpace("xgboost")
    basic = full.subset("basic")
    assert set(basic) < set(full)


def test_subset_needs_a_framework():
    """A hand-built space has no groups to narrow to."""
    with pytest.raises(ValueError, match="framework-built"):
        SearchSpace(knobs={"depth": IntRange(2, 6)}).subset("basic")


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_to_frame_matches_the_published_columns(framework):
    """to_frame is what Model.hpo_search_space() returns — pinned columns plus a JSON blob."""
    frame = SearchSpace(framework).to_frame()

    assert list(frame.columns) == ["knob", "default", "dist", "spec"]
    assert frame.notna().all().all()
    for blob in frame["spec"]:
        fields = json.loads(blob)
        assert "dist" not in fields and "default" not in fields  # those are pinned columns


class TestValidation:
    """Bad spaces fail at construction, not on trial 40."""

    def test_dist_is_required_and_not_inferred(self):
        with pytest.raises(ValueError, match="'dist'"):
            spec_from_dict({"low": 1, "high": 10})

    def test_unknown_dist(self):
        with pytest.raises(ValueError, match="'dist'"):
            spec_from_dict({"dist": "gaussian", "low": 1, "high": 10})

    def test_unknown_field_for_a_dist(self):
        with pytest.raises(ValueError, match="bad fields"):
            spec_from_dict({"dist": "int", "low": 1, "high": 10, "log": True})

    @pytest.mark.parametrize("cls", [IntRange, FloatRange])
    def test_inverted_range(self, cls):
        with pytest.raises(ValueError, match="low < high"):
            cls(10, 1)

    def test_log_scale_needs_a_positive_floor(self):
        with pytest.raises(ValueError, match="low > 0"):
            FloatRange(0.0, 1.0, log=True)

    def test_empty_choice(self):
        with pytest.raises(ValueError, match="at least one option"):
            Choice([])

    def test_a_default_outside_its_range_is_allowed(self):
        """Deliberate: the search explores away from where the untuned model sits, so a
        default need not lie inside the range being searched."""
        assert IntRange(2, 6, default=8).default == 8
