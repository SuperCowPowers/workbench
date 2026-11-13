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
    nca,
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


def test_curve_fold_error_reads_an_amplitude_shift():
    """Doubling F scales the curve, so the gap is 2x wherever both curves clear the floor.

    Only the sliver where the doubled curve is above the floor and the measured one is not
    reads less, so the score lands just under 2.
    """
    doubled = PARAMS.assign(f_percent=PARAMS["f_percent"] * 2)
    assert curve_fold_error(PARAMS, doubled).to_numpy() == pytest.approx(2.0, rel=0.05)
    assert (curve_fold_error(PARAMS, doubled) <= 2.0).all()


def test_curve_fold_error_caps_a_vanishing_prediction_at_the_floor():
    """A week-long measured half-life against a sub-hour prediction costs at most 1/floor."""
    slow = pd.DataFrame({"cl_ml_min_kg": [0.2], "vd_l_kg": [3.0], "f_percent": [40.0], "fu_percent": [10.0]})
    fast = slow.assign(cl_ml_min_kg=[40.0])
    for max_hours in (None, 24):
        score = curve_fold_error(slow, fast, max_hours=max_hours).iloc[0]
        assert 5 < score <= 100


def test_curve_fold_error_is_symmetric_in_a_rescaled_curve():
    """Over- and under-predicting by the same factor score the same."""
    assert curve_fold_error(PARAMS, PARAMS.assign(f_percent=PARAMS["f_percent"] * 3)).to_numpy() == pytest.approx(
        curve_fold_error(PARAMS.assign(f_percent=PARAMS["f_percent"] * 3), PARAMS).to_numpy(), rel=0.05
    )


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
            "cl_ml_min_kg": [20.0, 0.0, 20.0, 20.0, 80.0, 20.0, 20.0],
            "vd_l_kg": [2.0, 2.0, 0.01, 150.0, 2.0, 2.0, 2.0],
            "f_percent": [50.0, 50.0, 50.0, 50.0, 60.0, 120.0, 0.0],
            "fu_percent": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        },
        index=["ok", "zero_cl", "tiny_vd", "huge_vd", "first_pass", "f_over", "zero_f"],
    )
    flags = plausibility_flags(rows)
    assert not flags.loc["ok"].any()
    assert flags.loc["zero_cl", "cl_nonpositive"]
    assert flags.loc["tiny_vd", "vd_below_plasma"]
    assert flags.loc["huge_vd", "vd_high"] and not flags.loc["huge_vd", HARD_CHECKS].any()
    assert flags.loc["first_pass", "f_above_hepatic_ceiling"] and not flags.loc["first_pass", HARD_CHECKS].any()
    assert flags.loc["f_over", "f_above_hepatic_ceiling"] and not flags.loc["f_over", HARD_CHECKS].any()
    assert flags.loc["zero_f", "f_nonpositive"]


def _curve(curve_id, t, c, route="PO", blq=None):
    blq = np.zeros(len(t), dtype=bool) if blq is None else blq
    return pd.DataFrame({"curve_id": curve_id, "time_h": t, "conc": c, "blq": blq, "route": route})


def test_nca_recovers_a_bateman_curve():
    """Densely sampled one-compartment oral curve: NCA returns the analytic summaries."""
    ke, ka, f_dose, volume = 0.2, 1.5, 10.0, 2.0
    t = np.concatenate([np.linspace(0.05, 4, 80), np.linspace(4.5, 36, 64)])
    result = nca(_curve("a", t, bateman_concentration(t, f_dose, volume, ke, ka))).loc["a"]
    # the tie rule extends the fit toward the absorption phase, so the slope is close, not exact
    assert result["lambda_z"] == pytest.approx(ke, rel=1e-2)
    assert result["half_life"] == pytest.approx(np.log(2) / ke, rel=1e-2)
    assert result["auc_inf"] == pytest.approx(f_dose / (volume * ke), rel=1e-3)
    assert result["mrt"] == pytest.approx(1 / ke + 1 / ka, rel=1e-2)
    assert result["tmax"] == pytest.approx(bateman_tmax(ke, ka), abs=0.05)
    assert result["extrap_percent"] < 1.0


def test_nca_iv_curve_is_exact_and_starts_at_the_first_sample():
    """Mono-exponential IV decline: log-down trapezoids are exact and no t=0 point is added."""
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    c = 100.0 * np.exp(-0.3 * t)
    result = nca(_curve("iv", t, c, route="IV")).loc["iv"]
    assert result["auc_last"] == pytest.approx((c[0] - c[-1]) / 0.3)
    assert result["auc_inf"] == pytest.approx(c[0] / 0.3)
    assert result["lambda_z"] == pytest.approx(0.3)
    assert result["lambda_z_n"] == 5  # every sample after Cmax


def test_nca_blq_rule():
    """Leading BLQ counts as zero; embedded and trailing BLQ are dropped."""
    t = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
    c = np.array([0.0, 50.0, 80.0, 40.0, 0.0, 10.0, 5.0, 0.0])
    with_blq = nca(_curve("x", t, c, blq=c == 0)).loc["x"]
    keep = [0, 1, 2, 3, 5, 6]
    without = nca(_curve("x", t[keep], c[keep], blq=(c == 0)[keep])).loc["x"]
    assert with_blq["n_blq"] == 3
    assert with_blq["tlast"] == 12.0 and with_blq["clast"] == 5.0
    assert with_blq["auc_last"] == pytest.approx(without["auc_last"])
    # the leading zero contributes the 0.5 -> 1.0 triangle and nothing before it
    assert with_blq["auc_last"] > nca(_curve("x", t[1:7], c[1:7], route="IV")).loc["x", "auc_last"]


def test_nca_returns_nan_rather_than_guessing():
    """No quantifiable sample, too short a tail, or a rising tail leave the dependent columns NaN."""
    curves = pd.concat(
        [
            _curve("all_blq", [1.0, 2.0], [0.0, 0.0], blq=[True, True]),
            _curve("short", [1.0, 2.0, 4.0], [10.0, 20.0, 5.0]),
            _curve("rising", [1.0, 2.0, 4.0, 8.0, 12.0], [20.0, 1.0, 2.0, 3.0, 4.0]),
        ]
    )
    result = nca(curves)
    assert result.loc["all_blq", ["cmax", "auc_last"]].isna().all()
    assert result.loc["all_blq", "n_blq"] == 2
    assert result.loc["short", "auc_last"] > 0
    assert result.loc[["short", "rising"], ["lambda_z", "auc_inf", "half_life", "mrt"]].isna().all().all()
