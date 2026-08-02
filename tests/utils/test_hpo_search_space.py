"""Fast tests for get_hpo_search_space's framework dispatch (no AWS, no training deps).

The search-space modules under workbench.training defer their optuna/ray/framework
imports, so describing a space needs nothing installed beyond workbench itself — which is
what lets a lean environment (the dashboard) call Model.hpo_search_space().
"""

import json

import pandas as pd

# Workbench Imports
from workbench.core.artifacts.model_core import ModelFramework
from workbench.utils.training_job_utils import get_hpo_search_space


class _StubModel:
    """Just enough model for the dispatch: a framework."""

    def __init__(self, framework):
        self.model_framework = framework


def test_dispatch_accepts_the_enum_and_the_raw_string():
    """model_framework is an enum on a real Model, but a plain string in some meta reads."""
    by_enum = get_hpo_search_space(_StubModel(ModelFramework.XGBOOST))
    by_string = get_hpo_search_space(_StubModel("xgboost"))
    pd.testing.assert_frame_equal(by_enum, by_string)


def test_each_supported_framework_returns_its_own_space():
    """The three HPO frameworks each describe a different set of knobs."""
    spaces = {fw: set(get_hpo_search_space(_StubModel(fw))["knob"]) for fw in ("xgboost", "pytorch", "chemprop")}
    assert "max_depth" in spaces["xgboost"]
    assert "layers" in spaces["pytorch"]
    assert "ffn_hidden_dim" in spaces["chemprop"]
    # Distinct spaces, not one shared list
    assert spaces["xgboost"] != spaces["pytorch"] != spaces["chemprop"]


def test_unsupported_frameworks_return_none():
    """sklearn/meta/unknown have no search space — None rather than an empty frame, so the
    caller can tell 'not supported' from 'nothing to tune'."""
    for framework in (ModelFramework.SKLEARN, ModelFramework.META, ModelFramework.UNKNOWN, "transformer"):
        assert get_hpo_search_space(_StubModel(framework)) is None


def test_model_without_a_framework_returns_none():
    """A model that never resolved has no framework attribute at all — still None rather
    than an AttributeError."""

    class Unresolved:
        pass

    assert get_hpo_search_space(Unresolved()) is None


def test_returns_the_full_default_space():
    """Inspection always shows everything — every group the framework defines, which is
    also what a search explores when hpo["search_space"] is unset."""
    from workbench.training.xgb_hpo import xgb_search_space

    frame = get_hpo_search_space(_StubModel("xgboost"))
    assert set(frame["knob"]) == set(xgb_search_space())
    # Both groups, not just the first
    assert {"max_depth", "reg_alpha"} <= set(frame["knob"])


def test_columns_are_pinned_with_a_flexible_spec_blob():
    """Three pinned columns and one JSON blob — a knob's dist-specific fields never leak
    into another knob's row as NaN."""
    for framework in ("xgboost", "pytorch", "chemprop"):
        frame = get_hpo_search_space(_StubModel(framework))
        assert list(frame.columns) == ["knob", "default", "dist", "spec"]
        assert frame.notna().all().all()  # no holes anywhere
        for blob in frame["spec"]:
            assert isinstance(blob, str) and isinstance(json.loads(blob), dict)


def test_spec_blob_carries_the_fields_for_its_dist():
    """Each dist puts its own fields in the blob: bounds for a range, options for a
    categorical — and unset fields are dropped rather than serialized as null."""
    frame = get_hpo_search_space(_StubModel("chemprop")).set_index("knob")

    ints = json.loads(frame.loc["depth", "spec"])
    assert frame.loc["depth", "dist"] == "int"
    assert ints == {"low": 2, "high": 6, "step": 1}

    floats = json.loads(frame.loc["max_lr", "spec"])
    assert frame.loc["max_lr", "dist"] == "float"
    assert floats["log"] is True
    assert "step" not in floats  # unset, so absent rather than null

    choices = json.loads(frame.loc["ffn_hidden_dim", "spec"])
    assert frame.loc["ffn_hidden_dim", "dist"] == "choice"
    assert 300 in choices["options"]
    assert "low" not in choices


def test_default_keeps_each_knobs_native_type():
    """An int width stays an int and a shape stays a string — the column is not upcast to
    float, so a printed space reads as the values that actually train."""
    xgb = get_hpo_search_space(_StubModel("xgboost")).set_index("knob")
    assert xgb.loc["max_depth", "default"] == 7 and isinstance(xgb.loc["max_depth", "default"], int)
    assert isinstance(xgb.loc["learning_rate", "default"], float)

    pytorch = get_hpo_search_space(_StubModel("pytorch")).set_index("knob")
    assert pytorch.loc["layers", "default"] == "512-256-128"


def test_choice_options_are_scalars():
    """Options must be plottable as a categorical axis — no nested lists, which is what a
    tapered ffn shape would be if it weren't a dash-string."""
    for framework in ("chemprop", "pytorch"):
        frame = get_hpo_search_space(_StubModel(framework))
        for blob in frame.loc[frame["dist"] == "choice", "spec"]:
            options = json.loads(blob)["options"]
            assert all(isinstance(opt, (int, float, str)) for opt in options), options
