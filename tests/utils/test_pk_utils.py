"""Tests for the PK curve math in pk_utils."""

import numpy as np
import pandas as pd
import pytest

from workbench.utils.pk_utils import (
    HARD_CHECKS,
    auc_u_error_terms,
    bateman_concentration,
    bateman_tmax,
    curve_fold_error,
    exposure,
    plausibility_flags,
    unbound_concentration,
)

PARAMS = pd.DataFrame(
    {
        "cl_ml_min_kg": [20.0, 5.0, 60.0],
        "vd_l_kg": [2.0, 0.5, 8.0],
        "f_percent": [10.0, 80.0, 40.0],
        "fu_percent": [1.0, 20.0, 5.0],
    },
    index=["a", "b", "c"],
)


def test_bateman_broadcasts_like_the_scalar_form():
    """Array inputs, including a ka == ke element, match the per-compound scalar calls."""
    t = np.linspace(0, 24, 50)
    ke = np.array([0.5, 1.0, 2.0])
    stacked = bateman_concentration(t[None, :], 1.0, 1.0, ke[:, None], 1.0)
    for row, k in zip(stacked, ke):
        assert row == pytest.approx(bateman_concentration(t, 1.0, 1.0, k, 1.0))
    assert bateman_tmax(ke, 1.0)[1] == pytest.approx(1.0)


def test_auc_u_matches_the_integrated_curve():
    """auc_u = fu·F·Dose/CL must equal the area under the unbound curve."""
    t = np.linspace(0, 400, 400_000)
    area = np.trapezoid(unbound_concentration(PARAMS, t, dose=10.0), t, axis=1)
    assert area == pytest.approx(exposure(PARAMS, dose=10.0)["auc_u"].to_numpy(), rel=1e-4)


def test_cmax_u_is_the_curve_peak():
    t = np.linspace(0, 48, 200_000)
    curves = unbound_concentration(PARAMS, t)
    summary = exposure(PARAMS)
    assert curves.max(axis=1) == pytest.approx(summary["cmax_u"].to_numpy(), rel=1e-6)
    assert t[curves.argmax(axis=1)] == pytest.approx(summary["tmax"].to_numpy(), rel=1e-3)


def test_auc_u_ignores_vd_and_ka():
    wider = PARAMS.assign(vd_l_kg=PARAMS["vd_l_kg"] * 3)
    assert exposure(wider, ka=0.2)["auc_u"].to_numpy() == pytest.approx(exposure(PARAMS)["auc_u"].to_numpy())


def test_curve_fold_error_is_one_for_identical_curves():
    assert curve_fold_error(PARAMS, PARAMS).to_numpy() == pytest.approx(1.0)


def test_curve_fold_error_reads_an_amplitude_shift_exactly():
    """Doubling F scales the whole curve, so the log gap is log10(2) everywhere."""
    doubled = PARAMS.assign(f_percent=PARAMS["f_percent"] * 2)
    assert curve_fold_error(PARAMS, doubled).to_numpy() == pytest.approx(2.0)


def test_curve_fold_error_max_hours_bounds_a_long_tail():
    """A week-long measured half-life against a sub-hour prediction underflows uncapped."""
    slow = pd.DataFrame({"cl_ml_min_kg": [0.2], "vd_l_kg": [3.0], "f_percent": [40.0], "fu_percent": [10.0]})
    fast = slow.assign(cl_ml_min_kg=[40.0])
    assert np.isinf(curve_fold_error(slow, fast).iloc[0])
    capped = curve_fold_error(slow, fast, max_hours=24)
    assert np.isfinite(capped.iloc[0]) and capped.iloc[0] > 1
    doubled = PARAMS.assign(f_percent=PARAMS["f_percent"] * 2)
    assert curve_fold_error(PARAMS, doubled, max_hours=6).to_numpy() == pytest.approx(2.0)


def test_curve_fold_error_is_nan_for_a_nonpositive_prediction():
    broken = PARAMS.assign(f_percent=[10.0, -5.0, 40.0])
    result = curve_fold_error(PARAMS, broken)
    assert np.isnan(result["b"]) and not result.drop("b").isna().any()


def test_auc_u_error_terms_sum_to_the_auc_u_log_ratio():
    predicted = PARAMS.assign(cl_ml_min_kg=[24.0, 5.0, 30.0], f_percent=[25.0, 60.0, 40.0], vd_l_kg=[9.0, 1.0, 2.0])
    terms = auc_u_error_terms(PARAMS, predicted)
    ratio = exposure(predicted)["auc_u"] / exposure(PARAMS)["auc_u"]
    assert terms["auc_u"].to_numpy() == pytest.approx(np.log10(ratio).to_numpy())
    assert terms[["fu", "f", "cl"]].sum(axis=1).to_numpy() == pytest.approx(terms["auc_u"].to_numpy())
    assert terms.loc["c", "cl"] == pytest.approx(np.log10(2.0))


def test_plausibility_flags():
    rows = pd.DataFrame(
        {
            "cl_ml_min_kg": [20.0, 0.0, 20.0, 20.0, 80.0, 20.0],
            "vd_l_kg": [2.0, 2.0, 0.01, 150.0, 2.0, 2.0],
            "f_percent": [50.0, 50.0, 50.0, 50.0, 60.0, 120.0],
            "fu_percent": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        },
        index=["ok", "zero_cl", "tiny_vd", "huge_vd", "first_pass", "f_over"],
    )
    flags = plausibility_flags(rows)
    assert not flags.loc["ok"].any()
    assert flags.loc["zero_cl", "cl_nonpositive"]
    assert flags.loc["tiny_vd", "vd_below_plasma"]
    assert flags.loc["huge_vd", "vd_high"] and not flags.loc["huge_vd", HARD_CHECKS].any()
    assert flags.loc["first_pass", "f_above_hepatic_ceiling"] and not flags.loc["first_pass", HARD_CHECKS].any()
    assert flags.loc["f_over", "f_out_of_range"]
