"""Fast tests for get_hpo_importance's surrogate analysis (no AWS, no training deps).

The trials frame is the only input, so these build one directly rather than resolving a
model's S3 artifacts — which is also what lets a lean environment call
Model.hpo_importance().
"""

import json

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.utils import model_utils
from workbench.utils.model_utils import get_hpo_importance


def _trials(records, values, kind="trial"):
    """Build a trials frame in the shape get_hpo_results returns."""
    return pd.DataFrame(
        {
            "kind": [kind] * len(records),
            "value": values,
            "hyperparameters": [json.dumps(r) for r in records],
        }
    )


def _planted(n=60, noise=0.0):
    """A search where `signal` drives the objective and `dud` does nothing."""
    rng = np.random.default_rng(0)
    signal = rng.integers(1, 11, n)
    dud = rng.integers(1, 11, n)
    records = [{"signal": int(s), "dud": int(d)} for s, d in zip(signal, dud)]
    values = 1.0 + 0.05 * signal + noise * rng.standard_normal(n)
    return _trials(records, values.tolist())


@pytest.fixture
def stub_results(monkeypatch):
    """Point get_hpo_importance at a trials frame we control."""

    def _install(frame):
        monkeypatch.setattr(model_utils, "get_hpo_results", lambda model: {"trials": frame})

    return _install


def test_the_knob_that_drives_the_objective_ranks_first(stub_results):
    """A planted signal outranks a knob that does nothing, on both columns."""
    stub_results(_planted())
    frame = get_hpo_importance(object()).set_index("knob")
    assert frame.index[0] == "signal"
    assert frame.loc["signal", "importance"] > frame.loc["dud", "importance"]
    assert frame.loc["signal", "effect"] > frame.loc["dud", "effect"]


def test_best_points_the_right_way_for_a_minimized_objective(stub_results):
    """The objective rises with `signal`, so the best value is its floor, not its ceiling."""
    stub_results(_planted())
    frame = get_hpo_importance(object()).set_index("knob")
    assert frame.loc["signal", "best"] == 1


def test_importance_is_a_share_but_effect_is_absolute(stub_results):
    """The distinction the two columns exist for: importance always sums to 1, so in a
    search where nothing mattered a knob still looks important — effect is what says the
    total was negligible."""
    rng = np.random.default_rng(1)
    records = [{"a": int(x), "b": int(y)} for x, y in zip(rng.integers(1, 11, 60), rng.integers(1, 11, 60))]
    flat = _trials(records, (1.0 + 1e-6 * rng.standard_normal(60)).tolist())
    stub_results(flat)

    frame = get_hpo_importance(object())
    assert frame["importance"].sum() == pytest.approx(1.0)
    assert frame["importance"].max() > 0.2  # something still ranks high...
    assert frame["effect"].max() < 0.01  # ...on an objective that never moved


def test_categorical_knobs_report_their_own_values(stub_results):
    """A dash-string shape is ordinal-coded for the surrogate but reported as the string a
    user would actually set."""
    rng = np.random.default_rng(2)
    shapes = ["512-128", "1024-256-64", "256-64"]
    picks = rng.integers(0, 3, 60)
    records = [{"layers": shapes[p]} for p in picks]
    values = [1.0 + 0.1 * p for p in picks]  # first shape is best
    stub_results(_trials(records, values))

    frame = get_hpo_importance(object()).set_index("knob")
    assert frame.loc["layers", "best"] in shapes
    assert frame.loc["layers", "best"] == "512-128"


def test_a_knob_mixing_widths_and_shapes(stub_results):
    """ffn_hidden_dim holds scalar widths *and* tapered shapes in one knob, so the column
    is not numeric — it codes as a categorical and reports whichever cell won."""
    rng = np.random.default_rng(4)
    options = [900, 1200, 1800, "512-128", "1024-256-64"]
    picks = rng.integers(0, len(options), 60)
    records = [{"ffn_hidden_dim": options[p]} for p in picks]
    values = [1.0 + 0.1 * (p != 3) for p in picks]  # the tapered "512-128" is best
    stub_results(_trials(records, values))

    frame = get_hpo_importance(object()).set_index("knob")
    assert frame.loc["ffn_hidden_dim", "best"] == "512-128"
    assert frame.loc["ffn_hidden_dim", "effect"] > 1.0

    # A scalar width winning the same knob reports as 900, not "900"
    values = [1.0 + 0.1 * (p != 0) for p in picks]
    stub_results(_trials(records, values))
    best = get_hpo_importance(object()).set_index("knob").loc["ffn_hidden_dim", "best"]
    assert best == 900 and isinstance(best, int)


def test_int_knobs_keep_their_type(stub_results):
    """An int knob reports a width of 7, not 7.0 — the column is object, not upcast."""
    stub_results(_planted())
    frame = get_hpo_importance(object()).set_index("knob")
    assert isinstance(frame.loc["signal", "best"], int)


def test_a_knob_that_never_varied_scores_zero(stub_results):
    """One value means the search says nothing about it — zero rather than a NaN row."""
    rng = np.random.default_rng(3)
    varying = rng.integers(1, 11, 60)
    records = [{"pinned": 42, "varying": int(v)} for v in varying]
    stub_results(_trials(records, (1.0 + 0.05 * varying).tolist()))

    frame = get_hpo_importance(object()).set_index("knob")
    assert frame.loc["pinned", "importance"] == 0.0
    assert frame.loc["pinned", "effect"] == 0.0
    assert frame.loc["pinned", "best"] == 42


def test_baseline_row_is_excluded(stub_results):
    """The trials frame carries a `baseline` row scored on the same basis; it is not a
    trial and must not feed the surrogate."""
    trials = _planted(n=60)
    baseline = _trials([{"signal": 5, "dud": 5}], [99.0], kind="baseline")
    stub_results(pd.concat([trials, baseline], ignore_index=True))

    frame = get_hpo_importance(object())
    # The baseline's absurd value would blow up `effect` if it were included
    assert frame["effect"].max() < 100


def test_too_few_trials_returns_none(stub_results):
    """Under the minimum there is nothing to estimate from — None, not a garbage frame."""
    stub_results(_planted(n=5))
    assert get_hpo_importance(object()) is None


def test_unsearched_model_returns_none(monkeypatch):
    """None from get_hpo_results carries straight through."""
    monkeypatch.setattr(model_utils, "get_hpo_results", lambda model: None)
    assert get_hpo_importance(object()) is None
