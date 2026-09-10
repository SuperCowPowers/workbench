"""Tests for the Bateman PK plot."""

import numpy as np
import pytest

from workbench.utils.plots.pk import bateman, _concentration, _tmax

V, CL, F_PCT, DOSE = 50.0, 10.0, 40.0, 100.0
KE = CL / V
F_DOSE = DOSE * F_PCT / 100.0


def test_auc_matches_f_dose_over_clearance():
    """The integrated curve must equal F*D/CL -- the identity the annotation claims."""
    t = np.linspace(0, 40 * np.log(2) / KE, 200_000)
    for ka in (0.15, 0.5, 2.0, 5.0):
        auc = np.trapezoid(_concentration(t, F_DOSE, V, KE, ka), t)
        assert auc == pytest.approx(F_DOSE / CL, rel=1e-4)


def test_tmax_is_the_curve_peak():
    """Closed-form Tmax must agree with where the sampled curve actually peaks."""
    for ka in (0.15, 0.5, 2.0, 5.0):
        peak = _tmax(KE, ka)
        t = np.linspace(0, 4 * peak, 400_001)
        assert t[np.argmax(_concentration(t, F_DOSE, V, KE, ka))] == pytest.approx(peak, rel=1e-3)


def test_ka_equals_ke_limit_is_continuous():
    """ka == ke divides by zero in the general form; the limit form must join smoothly.

    Continuity shows up as the gap to the limit shrinking in step with the offset, so
    assert that scaling rather than picking an epsilon out of the air.
    """
    t = np.linspace(0, 30, 200)
    at_limit = _concentration(t, F_DOSE, V, KE, KE)
    assert np.all(np.isfinite(at_limit))

    gaps = [np.max(np.abs(_concentration(t, F_DOSE, V, KE, KE + off) - at_limit)) for off in (1e-4, 1e-5, 1e-6)]
    assert gaps[0] == pytest.approx(10 * gaps[1], rel=1e-2)
    assert gaps[1] == pytest.approx(10 * gaps[2], rel=1e-2)
    assert gaps[-1] < 1e-5

    # Approaching from below must land on the same limit.
    assert _concentration(t, F_DOSE, V, KE, KE - 1e-6) == pytest.approx(at_limit, abs=1e-5)
    assert _tmax(KE, KE) == pytest.approx(_tmax(KE, KE + 1e-6), rel=1e-4)


def test_concentration_starts_at_zero_and_decays():
    """An oral profile rises from nothing and comes back down."""
    t = np.linspace(0, 60, 1000)
    conc = _concentration(t, F_DOSE, V, KE, 1.0)
    assert conc[0] == pytest.approx(0.0)
    assert np.all(conc >= 0)
    assert conc[-1] < conc.max() / 100


def test_flip_flop_absorption_lowers_the_peak():
    """Slower absorption than elimination flattens the curve without changing exposure."""
    fast = _concentration(np.linspace(0, 60, 5000), F_DOSE, V, KE, 5.0).max()
    slow = _concentration(np.linspace(0, 60, 5000), F_DOSE, V, KE, 0.05).max()
    assert slow < fast


def test_figure_has_one_visible_pair_per_slider_step():
    fig = bateman(volume=V, clearance=CL, f_percent=F_PCT, dose=DOSE, ka_steps=12)
    assert len(fig.data) == 24
    assert len(fig.layout.sliders[0].steps) == 12
    assert sum(bool(trace.visible) for trace in fig.data) == 2
    for step in fig.layout.sliders[0].steps:
        assert sum(step.args[0]["visible"]) == 2


def test_slider_steps_carry_their_own_readout():
    fig = bateman(volume=V, clearance=CL, ka_steps=6)
    texts = [step.args[1]["annotations"][0]["text"] for step in fig.layout.sliders[0].steps]
    assert len(set(texts)) == 6
    assert "flip-flop" in texts[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume": 0},
        {"clearance": -1},
        {"f_percent": 0},
        {"f_percent": 101},
        {"dose": 0},
        {"ka_range": (5.0, 0.1)},
        {"ka_range": (0, 1.0)},
    ],
)
def test_invalid_inputs_raise(kwargs):
    args = {"volume": V, "clearance": CL, **kwargs}
    with pytest.raises(ValueError):
        bateman(**args)


def test_default_window_covers_the_slowest_slider_position():
    """The window is shared across ka, so the slowest ka must not run off the right edge."""
    fig = bateman(volume=V, clearance=CL, f_percent=F_PCT, dose=DOSE, ka_range=(0.1, 5.0))
    window = fig.data[0].x[-1]
    t = np.linspace(0, window, 50_000)
    for ka in (0.1, 0.2, 1.0, 5.0):
        captured = np.trapezoid(_concentration(t, F_DOSE, V, KE, ka), t)
        assert captured / (F_DOSE / CL) > 0.9
