"""Fast tests for the HPO parallel-coordinates plot (no AWS, no matplotlib display).

The plot reads a model only through `hpo_results()`, `hpo_search_space()` and
`hpo_importance()`, so a stub standing in for those covers it without resolving S3
artifacts. The assertions are about what the colours *mean*, since that is the part a
refactor can silently invert.
"""

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

# Workbench Imports
from workbench.utils.hpo_plots import hpo_parallel_coordinates  # noqa: E402


class _StubModel:
    """A searched model, as far as the plot is concerned."""

    name = "stub-hpo"

    def __init__(self, trials, *, baseline=0.50, best=0.40, space=True):
        self._trials = trials
        self._baseline, self._best = baseline, best
        self._space = space

    def hpo_results(self):
        return {
            "metric": "cv_mae",
            "trials": self._trials,
            "best_config": {"depth": 6, "rate": 0.01},
            "search_best_value": self._best,
            "search_baseline_value": self._baseline,
            "trial_counts": {"attempted": len(self._trials), "completed": 3, "pruned": 2, "failed": 0},
        }

    def hpo_search_space(self):
        if not self._space:
            return None
        return pd.DataFrame(
            {
                "knob": ["depth", "rate"],
                "dist": ["int", "float"],
                "spec": [json.dumps({"low": 2, "high": 8}), json.dumps({"low": 0.001, "high": 0.1, "log": True})],
            }
        )

    def hpo_importance(self):
        return pd.DataFrame({"knob": ["rate", "depth"], "importance": [0.8, 0.2], "effect": [3.0, 1.0]})


def _frame(rows):
    """A trials frame in the shape get_hpo_results returns."""
    return pd.DataFrame(
        [
            {
                "number": i,
                "value": value,
                "step": step,
                "completed": completed,
                "kind": kind,
                "hyperparameters": json.dumps({"depth": depth, "rate": rate}),
            }
            for i, (value, step, completed, kind, depth, rate) in enumerate(rows)
        ]
    )


def _default_rows():
    """Two stopped trials scoring *better* than every completed one -- the aqsol shape, where
    an easy first fold makes a culled trial look like a winner."""
    return [
        (0.50, 5, True, "baseline", 4, 0.02),
        (0.40, 5, True, "trial", 6, 0.01),
        (0.46, 5, True, "trial", 3, 0.05),
        (0.30, 1, False, "trial", 8, 0.09),  # stopped at the first rung, flattering value
        (0.34, 2, False, "trial", 2, 0.001),
    ]


# What each early fold understates the full ensemble by, planted so the offsets are known.
_PLANTED = {1: 0.04, 2: 0.02, 4: 0.01}


def _with_trajectories(frame, full_step=5):
    """Give every row a trajectory whose early folds sit below its own value by a known gap.

    Only the completed rows feed the offsets, but the stopped ones carry a history too, the
    same way a real record does.
    """
    trajectories = []
    for _, row in frame.iterrows():
        history = {fold: row["value"] - gap for fold, gap in _PLANTED.items() if fold <= row["step"]}
        history[int(row["step"])] = row["value"]  # the trial's own value sits at where it ended
        trajectories.append(json.dumps({str(fold): value for fold, value in sorted(history.items())}))
    frame = frame.copy()
    frame["trajectory"] = trajectories
    return frame


def _crowd(fig):
    """The trial lines, which sit between the axis rules (zorder 0) and the references (10)."""
    return [line for line in fig.axes[0].lines if line.get_zorder() in (1, 2)]


def _rgb(line):
    """A line's colour as a comparable triple -- it may be a hex string or an RGBA tuple."""
    from matplotlib.colors import to_rgb

    return tuple(round(channel, 6) for channel in to_rgb(line.get_color()))


def test_a_stopped_trial_stays_off_the_scale_without_a_trajectory():
    """Its objective covers fewer members, and on some datasets that reads better than a full
    run -- colouring it by the raw value would paint the search's rejects as its winners. With
    nothing to estimate the shortfall from, grey is the honest answer."""
    fig = hpo_parallel_coordinates(_StubModel(_frame(_default_rows())))
    colours = [_rgb(line) for line in _crowd(fig)]

    # Greys are r == g == b; the diverging colormap's entries are not. Both stopped trials
    # take the same one, so nothing about them reads as a position on the objective scale.
    greys = [c for c in colours if len(set(c)) == 1]
    assert len(greys) == 2 and len(set(greys)) == 1
    assert len(set(colours)) == 3  # two completed, each its own hue, plus the one grey


def test_fold_offsets_recover_a_planted_shortfall():
    """The estimator behind the estimate: what fold *k* understates the full ensemble by,
    read off the completed trials that carry both ends of a trajectory."""
    from workbench.utils.hpo_plots import _fold_offsets

    frame = _with_trajectories(_frame(_default_rows()))
    completed = frame["completed"].astype(bool)

    assert _fold_offsets(frame, completed, 5) == pytest.approx(_PLANTED)


def test_a_trajectory_carries_a_stopped_trial_onto_the_scale():
    """With the shortfall known, a stopped trial gets a hue instead of a grey -- and the
    legend says the hue is an estimate, since that is all that separates it from a measured
    one."""
    frame = _with_trajectories(_frame(_default_rows()))
    fig = hpo_parallel_coordinates(_StubModel(frame))

    assert not [line for line in _crowd(fig) if len(set(_rgb(line))) == 1], "no trial should be grey"
    caveat = fig.axes[0].get_legend().get_texts()[-1]
    assert "estimates" in caveat.get_text() and caveat.get_style() == "italic"


def test_a_trajectory_with_no_usable_step_estimates_nothing():
    """A frame can carry trajectories and still have no step to key them on -- an objective
    that reports nothing leaves the column empty. NaN is truthy, so the guard has to test for
    it rather than for falsiness."""
    frame = _with_trajectories(_frame(_default_rows()))
    frame["step"] = float("nan")
    frame["completed"] = True

    fig = hpo_parallel_coordinates(_StubModel(frame))
    assert fig is not None


def test_a_fold_no_completed_trial_reported_at_leaves_its_trials_grey():
    """The offsets only cover the folds a completed trajectory passed through. A trial stopped
    anywhere else has nothing to be carried by, and is not guessed at."""
    rows = _default_rows() + [(0.32, 3, False, "trial", 5, 0.03)]  # nothing completed reports fold 3
    fig = hpo_parallel_coordinates(_StubModel(_with_trajectories(_frame(rows))))

    greys = [line for line in _crowd(fig) if len(set(_rgb(line))) == 1]
    assert len(greys) == 1


def test_the_colorbar_is_the_metric_and_is_centred_on_the_baseline():
    """Ticks read as MAE rather than a margin, and the divergence point is the baseline, so
    hue still answers 'did this beat my defaults'."""
    fig = hpo_parallel_coordinates(_StubModel(_frame(_default_rows())))
    bar = fig.axes[-1]
    rules = [line.get_ydata()[0] for line in bar.lines]

    assert 0.50 in rules and 0.40 in rules  # baseline and published, in raw objective units
    low, high = bar.get_ylim()
    assert low < 0.40 and high > 0.50


def test_a_search_where_nothing_beat_the_baseline_still_scales():
    """Scaling by the best margin collapses to a point when there is no margin, which would
    paint every trial as the baseline."""
    rows = [(0.50, 5, True, "baseline", 4, 0.02), (0.62, 5, True, "trial", 6, 0.01)]
    fig = hpo_parallel_coordinates(_StubModel(_frame(rows), best=0.62))
    low, high = fig.axes[-1].get_ylim()
    assert high > low


def test_an_artifact_with_no_step_column_still_plots():
    """Runs recorded before the ladder carry no `step`; the hover loses where a trial stopped,
    not the plot."""
    frame = _frame(_default_rows()).drop(columns=["step"])
    fig = hpo_parallel_coordinates(_StubModel(frame))
    assert len(_crowd(fig)) == 4  # the baseline row is a reference line, not one of the crowd


def test_an_unsearched_model_returns_none():
    class _Plain:
        name = "plain"

        def hpo_results(self):
            return None

    assert hpo_parallel_coordinates(_Plain()) is None


@pytest.mark.parametrize("space", [True, False])
def test_a_missing_search_space_falls_back_to_observed_ranges(space):
    """The space scales the axes to what the search *could* have explored; without one the
    trials' own range has to do."""
    assert hpo_parallel_coordinates(_StubModel(_frame(_default_rows()), space=space)) is not None
