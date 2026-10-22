"""Pharmacokinetic curve math: the one-compartment oral (Bateman) model and the
exposure summaries used to score predicted PK parameters against measured ones.

Parameter frames carry one row per compound and these columns:

    cl_ml_min_kg   total plasma clearance, mL/min/kg
    vd_l_kg        volume of distribution, L/kg
    f_percent      oral bioavailability, %
    fu_percent     plasma fraction unbound, % free

Dose is in mg/kg, so concentrations read in mg/L and AUC in mg·h/L. Fold-error
comparisons between two parameter frames do not depend on dose.
"""

import numpy as np
import pandas as pd

PK_PARAMS = ["cl_ml_min_kg", "vd_l_kg", "f_percent", "fu_percent"]

# mL/min/kg -> L/h/kg. CL and Vd are reported in units that do not agree.
CL_TO_L_PER_H = 0.06

# Absorption rate (1/h) assumed when none is measured. Nothing in an early ADMET package pins it down.
DEFAULT_KA = 1.0

# Hepatic blood flow, mL/min/kg (Davies & Morris 1993). Caps F for hepatically cleared compounds.
HEPATIC_BLOOD_FLOW = {"mouse": 90.0, "rat": 55.0, "monkey": 44.0, "dog": 31.0, "human": 21.0}

# Plausibility checks: HARD_CHECKS are impossible values, the rest are flags for review.
HARD_CHECKS = ["f_out_of_range", "fu_out_of_range", "cl_nonpositive", "vd_below_plasma"]


def bateman_concentration(t, f_dose, volume, ke, ka):
    """Bateman one-compartment oral profile, mass units per volume unit. Broadcasts over all arguments."""
    t, ke, ka = np.asarray(t, dtype=float), np.asarray(ke, dtype=float), np.asarray(ka, dtype=float)
    # ka == ke is a removable singularity; the limit is the t*exp(-ke*t) form.
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (f_dose * ka) / (volume * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    limit = (f_dose * ke / volume) * t * np.exp(-ke * t)
    return np.where(np.abs(ka - ke) < 1e-9, limit, general)[()]


def bateman_tmax(ke, ka):
    """Time of peak concentration. Broadcasts over both arguments."""
    ke, ka = np.asarray(ke, dtype=float), np.asarray(ka, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        general = np.log(ka / ke) / (ka - ke)
    return np.where(np.abs(ka - ke) < 1e-9, 1.0 / ke, general)[()]


def elimination_rate(params: pd.DataFrame) -> np.ndarray:
    """Elimination rate constant ke = CL / Vd per compound, 1/h."""
    return CL_TO_L_PER_H * params["cl_ml_min_kg"].to_numpy(float) / params["vd_l_kg"].to_numpy(float)


def unbound_concentration(params: pd.DataFrame, t, dose: float = 1.0, ka: float = DEFAULT_KA) -> np.ndarray:
    """Unbound plasma concentration after a single oral dose.

    Args:
        params (pd.DataFrame): PK parameter frame (see module docstring).
        t (array-like): Hours. A 1-D grid shared by every compound, or one row per compound.
        dose (float): Oral dose, mg/kg. Defaults to 1.0.
        ka (float): Absorption rate constant, 1/h. Defaults to DEFAULT_KA.

    Returns:
        np.ndarray: Concentration in mg/L, one row per compound, one column per time point.
    """
    t = np.asarray(t, dtype=float)
    t = t[None, :] if t.ndim == 1 else t
    column = lambda name: params[name].to_numpy(float)[:, None]  # noqa: E731
    f_dose = dose * column("f_percent") / 100.0
    conc = bateman_concentration(t, f_dose, column("vd_l_kg"), elimination_rate(params)[:, None], ka)
    return column("fu_percent") / 100.0 * conc


def exposure(params: pd.DataFrame, dose: float = 1.0, ka: float = DEFAULT_KA) -> pd.DataFrame:
    """Unbound exposure summaries per compound after a single oral dose.

    `auc_u` is fu·F·Dose/CL and depends on neither ka nor Vd. `cmax_u` and `tmax` depend on
    ka, so they partly score the assumed absorption rate.

    Args:
        params (pd.DataFrame): PK parameter frame (see module docstring).
        dose (float): Oral dose, mg/kg. Defaults to 1.0.
        ka (float): Absorption rate constant, 1/h. Defaults to DEFAULT_KA.

    Returns:
        pd.DataFrame: auc_u (mg·h/L), half_life (h), tmax (h) and cmax_u (mg/L), indexed like `params`.
    """
    ke = elimination_rate(params)
    fu = params["fu_percent"].to_numpy(float) / 100.0
    f_dose = dose * params["f_percent"].to_numpy(float) / 100.0
    tmax = bateman_tmax(ke, ka)
    return pd.DataFrame(
        {
            "auc_u": fu * f_dose / (CL_TO_L_PER_H * params["cl_ml_min_kg"].to_numpy(float)),
            "half_life": np.log(2) / ke,
            "tmax": tmax,
            "cmax_u": fu * bateman_concentration(tmax, f_dose, params["vd_l_kg"].to_numpy(float), ke, ka),
        },
        index=params.index,
    )


def curve_fold_error(
    measured: pd.DataFrame,
    predicted: pd.DataFrame,
    ka: float = DEFAULT_KA,
    floor: float = 0.01,
    max_hours: float = None,
    n_points: int = 500,
) -> pd.Series:
    """Whole-curve agreement per compound: 10^mean|log10 Cu_pred(t) − log10 Cu_obs(t)|.

    Reads as an average fold error over the unbound curve. Both curves are clipped at a
    quantitation floor, `floor` × measured Cmax, the way an assay reports anything below
    its limit as not measurable, and the gap is averaged where either curve is above it.
    A predicted curve that vanishes while the measured one persists therefore costs at
    most 1/`floor`, rather than a log gap that grows without bound down the tail.

    Args:
        measured (pd.DataFrame): Measured PK parameter frame.
        predicted (pd.DataFrame): Predicted PK parameter frame, same index as `measured`.
        ka (float): Absorption rate constant shared by both curves, 1/h. Defaults to DEFAULT_KA.
        floor (float): Quantitation floor as a fraction of measured Cmax. Defaults to 0.01.
        max_hours (float, optional): Last time point, e.g. a study's last sample, h. Defaults to no cap.
        n_points (int): Time samples per compound. Defaults to 500.

    Returns:
        pd.Series: Fold error per compound (1.0 is a perfect overlay), indexed like `measured`;
            NaN where a predicted parameter is not positive.
    """
    valid = (predicted[PK_PARAMS] > 0).all(axis=1).to_numpy()

    # Ten half-lives of the slower process past each curve's peak puts both floors inside the grid
    with np.errstate(divide="ignore", invalid="ignore"):
        horizon = np.fmax(
            *(
                bateman_tmax(ke, ka) + 10 * np.log(2) / np.minimum(ke, ka)
                for ke in (elimination_rate(measured), elimination_rate(predicted))
            )
        )
    if max_hours is not None:
        horizon = np.minimum(horizon, max_hours)
    t = horizon[:, None] * np.linspace(0.0, 1.0, n_points)[None, :]

    obs = unbound_concentration(measured, t, ka=ka)
    pred = unbound_concentration(predicted, t, ka=ka)
    lloq = floor * unbound_concentration(measured, bateman_tmax(elimination_rate(measured), ka)[:, None], ka=ka)
    in_window = (obs >= lloq) | (pred >= lloq)
    gap = np.abs(np.log10(np.maximum(pred, lloq)) - np.log10(np.maximum(obs, lloq)))
    fold = 10 ** np.mean(gap, axis=1, where=in_window)
    return pd.Series(np.where(valid, fold, np.nan), index=measured.index)


def auc_u_error_terms(measured: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Split each compound's log10(AUCu_pred / AUCu_obs) into per-parameter contributions.

    AUCu = fu·F·Dose/CL is a product, so its log error is exactly the sum of the
    `fu`, `f` and `cl` columns. `cl` is negated: over-predicting clearance lowers AUCu.
    Vd and ka do not enter.

    Args:
        measured (pd.DataFrame): Measured PK parameter frame.
        predicted (pd.DataFrame): Predicted PK parameter frame, same index as `measured`.

    Returns:
        pd.DataFrame: log10 contributions `fu`, `f`, `cl` and their sum `auc_u`.
    """

    def log_ratio(name):
        return np.log10(predicted[name] / measured[name])

    terms = pd.DataFrame({"fu": log_ratio("fu_percent"), "f": log_ratio("f_percent"), "cl": -log_ratio("cl_ml_min_kg")})
    terms["auc_u"] = terms.sum(axis=1)
    return terms


def plausibility_flags(params: pd.DataFrame, species: str = "mouse") -> pd.DataFrame:
    """Physical-plausibility checks per compound; True marks a violation.

    Columns in HARD_CHECKS are impossible values (a defect in whatever produced them).
    The rest are flags for review:

    - `vd_high`: Vd above 100 L/kg, real for some lipophilic bases.
    - `half_life_out_of_range`: t½ outside 5 minutes to 1 week.
    - `f_above_hepatic_ceiling`: F above 100·(1 − CL/Q_h), the first-pass cap for a
      hepatically cleared compound. Renal and biliary clearance are legitimate exceptions.

    Args:
        params (pd.DataFrame): PK parameter frame (see module docstring).
        species (str): Key into HEPATIC_BLOOD_FLOW. Defaults to "mouse".

    Returns:
        pd.DataFrame: One boolean column per check, indexed like `params`.
    """
    cl, vd = params["cl_ml_min_kg"], params["vd_l_kg"]
    f, fu = params["f_percent"], params["fu_percent"]
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = np.log(2) * vd / (CL_TO_L_PER_H * cl)
    f_ceiling = 100.0 * (1.0 - cl / HEPATIC_BLOOD_FLOW[species])
    return pd.DataFrame(
        {
            "f_out_of_range": ~f.between(0, 100, inclusive="right"),
            "fu_out_of_range": ~fu.between(0, 100, inclusive="right"),
            "cl_nonpositive": cl <= 0,
            "vd_below_plasma": vd < 0.04,
            "vd_high": vd > 100,
            "half_life_out_of_range": ~half_life.between(5 / 60, 7 * 24),
            "f_above_hepatic_ceiling": f > f_ceiling,
        },
        index=params.index,
    )
