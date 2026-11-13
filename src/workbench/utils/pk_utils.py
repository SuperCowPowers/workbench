"""Pharmacokinetic curve math: the one-compartment oral (Bateman) model, the exposure
summaries used to score predicted PK parameters against measured ones, and
non-compartmental analysis (NCA) of measured concentration-time curves.

Parameter frames carry one row per compound and these columns:

    cl_ml_min_kg   total plasma clearance, mL/min/kg
    vss_l_kg        volume of distribution, L/kg
    f_percent      oral bioavailability, %
    fu_percent     plasma fraction unbound, % free

Dose is in mg/kg, so concentrations read in mg/L and AUC in mg·h/L. Fold-error
comparisons between two parameter frames do not depend on dose.
"""

import numpy as np
import pandas as pd

PK_PARAMS = ["cl_ml_min_kg", "vss_l_kg", "f_percent", "fu_percent"]

# mL/min/kg -> L/h/kg. CL and Vss are reported in units that do not agree.
CL_TO_L_PER_H = 0.06

# Absorption rate (1/h) assumed when none is measured. Nothing in an early ADMET package pins it down.
DEFAULT_KA = 1.0

# Hepatic blood flow, mL/min/kg (Davies & Morris 1993). Caps F for hepatically cleared compounds.
HEPATIC_BLOOD_FLOW = {"mouse": 90.0, "rat": 55.0, "monkey": 44.0, "dog": 31.0, "human": 21.0}

# Plausibility checks: HARD_CHECKS are impossible values, the rest are flags for review.
HARD_CHECKS = ["f_nonpositive", "fu_out_of_range", "cl_nonpositive", "vss_below_plasma"]

# Routes dosed straight into circulation: concentration at t=0 is not zero.
IV_ROUTES = {"IV"}

# Terminal-slope fits within this adjusted R² of the best one count as tied; the longest wins.
LAMBDA_Z_R2_TOLERANCE = 1e-4

NCA_COLUMNS = [
    "n_points",
    "n_blq",
    "cmax",
    "tmax",
    "tlast",
    "clast",
    "auc_last",
    "auc_inf",
    "extrap_percent",
    "lambda_z",
    "lambda_z_n",
    "lambda_z_r2",
    "half_life",
    "aumc_inf",
    "mrt",
]

NCA_PARAM_COLUMNS = [
    "cmax_ng_ml",
    "auc_last_ng_h_ml",
    "auc_inf_ng_h_ml",
    "cl_ml_min_kg",
    "vss_l_kg",
    "vz_l_kg",
    "f_percent",
]


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
    """Elimination rate constant ke = CL / Vss per compound, 1/h."""
    return CL_TO_L_PER_H * params["cl_ml_min_kg"].to_numpy(float) / params["vss_l_kg"].to_numpy(float)


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
    conc = bateman_concentration(t, f_dose, column("vss_l_kg"), elimination_rate(params)[:, None], ka)
    return column("fu_percent") / 100.0 * conc


def exposure(params: pd.DataFrame, dose: float = 1.0, ka: float = DEFAULT_KA) -> pd.DataFrame:
    """Unbound exposure summaries per compound after a single oral dose.

    `auc_u` is fu·F·Dose/CL and depends on neither ka nor Vss. `cmax_u` and `tmax` depend on
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
            "cmax_u": fu * bateman_concentration(tmax, f_dose, params["vss_l_kg"].to_numpy(float), ke, ka),
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
    Vss and ka do not enter.

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

    - `vss_high`: Vss above 100 L/kg, real for some lipophilic bases.
    - `half_life_out_of_range`: t½ outside 5 minutes to 1 week.
    - `f_above_hepatic_ceiling`: F above 100·(1 − CL/Q_h), the first-pass cap for a
      hepatically cleared compound. Renal and biliary clearance are legitimate exceptions.
      Measured F above 100% lands here too: dose-normalized F% from PO doses well above
      the IV dose inflates by CL_IV/CL_PO when clearance saturates.

    Args:
        params (pd.DataFrame): PK parameter frame (see module docstring).
        species (str): Key into HEPATIC_BLOOD_FLOW. Defaults to "mouse".

    Returns:
        pd.DataFrame: One boolean column per check, indexed like `params`.
    """
    cl, vss = params["cl_ml_min_kg"], params["vss_l_kg"]
    f, fu = params["f_percent"], params["fu_percent"]
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = np.log(2) * vss / (CL_TO_L_PER_H * cl)
    f_ceiling = 100.0 * (1.0 - cl / HEPATIC_BLOOD_FLOW[species])
    return pd.DataFrame(
        {
            "f_nonpositive": f <= 0,
            "fu_out_of_range": ~fu.between(0, 100, inclusive="right"),
            "cl_nonpositive": cl <= 0,
            "vss_below_plasma": vss < 0.04,
            "vss_high": vss > 100,
            "half_life_out_of_range": ~half_life.between(5 / 60, 7 * 24),
            "f_above_hepatic_ceiling": f > f_ceiling,
        },
        index=params.index,
    )


def _terminal_slope(t: np.ndarray, c: np.ndarray) -> tuple:
    """Best log-linear fit over the last 3+ points: (lambda_z, n points, adjusted R²), NaNs if none declines."""
    best = (np.nan, np.nan, np.nan)
    log_c = np.log(c)
    for n in range(3, len(t) + 1):
        x, y = t[-n:], log_c[-n:]
        slope, intercept = np.polyfit(x, y, 1)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if slope >= 0 or ss_tot == 0:
            continue
        r2 = 1.0 - np.sum((y - (slope * x + intercept)) ** 2) / ss_tot
        adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
        if np.isnan(best[2]) or adjusted >= best[2] - LAMBDA_Z_R2_TOLERANCE:
            best = (-slope, n, adjusted)
    return best


def _nca_curve(t: np.ndarray, c: np.ndarray, blq: np.ndarray, start: str) -> dict:
    """NCA summaries for one curve; `t` is sorted and free of missing concentrations."""
    result = dict.fromkeys(NCA_COLUMNS, np.nan)
    result["n_points"], result["n_blq"] = len(t), int(blq.sum())
    quantifiable = np.flatnonzero(~blq)
    if len(quantifiable) == 0:
        return result

    # BLQ rule: zero before the first quantifiable sample, dropped when embedded or trailing
    keep = ~blq
    keep[: quantifiable[0]] = True
    c = np.where(blq, 0.0, c)
    t, c = t[keep], c[keep]
    peak = int(np.argmax(c))
    tlast, clast = t[-1], c[-1]
    result.update(cmax=c[peak], tmax=t[peak], tlast=tlast, clast=clast)
    terminal_t, terminal_c = t[peak + 1 :], c[peak + 1 :]

    # Concentration at t=0: zero for an extravascular dose, back-extrapolated C0 for an IV bolus
    if start is not None and t[0] > 0:
        c0 = 0.0
        if start == "iv":
            log_linear = len(c) > 1 and c[0] > c[1] > 0
            c0 = c[0] * (c[0] / c[1]) ** (t[0] / (t[1] - t[0])) if log_linear else c[0]
        t, c = np.insert(t, 0, 0.0), np.insert(c, 0, c0)

    # Linear-up / log-down trapezoids for AUC and its first moment
    t1, t2, c1, c2 = t[:-1], t[1:], c[:-1], c[1:]
    dt = t2 - t1
    declining = (c2 < c1) & (c2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(declining, np.log(c1 / c2), 1.0)
    auc = np.where(declining, (c1 - c2) * dt / log_ratio, dt * (c1 + c2) / 2.0)
    aumc = np.where(
        declining,
        (t1 * c1 - t2 * c2) * dt / log_ratio + (c1 - c2) * dt**2 / log_ratio**2,
        dt * (t1 * c1 + t2 * c2) / 2.0,
    )
    result["auc_last"] = auc.sum()

    # Terminal phase: samples after the peak
    lambda_z, n, r2 = _terminal_slope(terminal_t, terminal_c) if len(terminal_t) >= 3 else (np.nan,) * 3
    if np.isnan(lambda_z):
        return result
    auc_inf = result["auc_last"] + clast / lambda_z
    aumc_inf = aumc.sum() + clast * tlast / lambda_z + clast / lambda_z**2
    result.update(
        auc_inf=auc_inf,
        extrap_percent=100.0 * (clast / lambda_z) / auc_inf,
        lambda_z=lambda_z,
        lambda_z_n=n,
        lambda_z_r2=r2,
        half_life=np.log(2) / lambda_z,
        aumc_inf=aumc_inf,
        mrt=aumc_inf / auc_inf,
    )
    return result


def nca(
    curves: pd.DataFrame,
    curve: str = "curve_id",
    time: str = "time_h",
    conc: str = "conc",
    blq: str = "blq",
    route: str = "route",
) -> pd.DataFrame:
    """Non-compartmental analysis of measured concentration-time curves, one row per curve.

    Unit-agnostic: areas come back in the input's concentration × time units. Conventions
    are the Phoenix WinNonlin defaults:

    - AUC by linear-up / log-down trapezoids.
    - BLQ samples (flagged, or a non-positive concentration) count as zero before the first
      quantifiable sample and are dropped when embedded or trailing.
    - lambda_z is the log-linear fit over the last 3+ samples after Cmax with the best
      adjusted R², ties going to the longer fit. `lambda_z_r2` is that adjusted R².
    - A curve with no t=0 sample gets one: zero for an extravascular dose, and for an IV
      bolus C0 back-extrapolated log-linearly through the first two samples (the first
      sample's value when those two do not decline). `cmax` and `tmax` stay observed values.

    Summaries that cannot be computed are NaN: everything past the counts for a curve with
    no quantifiable sample, and everything derived from lambda_z when fewer than 3 samples
    follow Cmax or none of the fits declines.

    Args:
        curves (pd.DataFrame): One row per sample.
        curve (str): Column identifying the curve. Defaults to "curve_id".
        time (str): Sample time column. Defaults to "time_h".
        conc (str): Concentration column. Defaults to "conc".
        blq (str): Boolean below-quantitation column, or None for none. Defaults to "blq".
        route (str): Dosing route column, or None to start every curve at its first sample.
            Defaults to "route".

    Returns:
        pd.DataFrame: NCA_COLUMNS, indexed by `curve`.
    """
    curves = curves.dropna(subset=[time]).sort_values(time)
    rows = {}
    for curve_id, group in curves.groupby(curve, sort=True):
        c = group[conc].to_numpy(float, na_value=np.nan)
        flagged = group[blq].fillna(False).to_numpy(bool) if blq else np.zeros(len(group), dtype=bool)
        measured = flagged | ~np.isnan(c)
        below = (flagged | (np.nan_to_num(c) <= 0))[measured]
        start = None if route is None else "iv" if group[route].isin(IV_ROUTES).any() else "extravascular"
        rows[curve_id] = _nca_curve(group[time].to_numpy(float)[measured], c[measured], below, start)
    return pd.DataFrame.from_dict(rows, orient="index", columns=NCA_COLUMNS).rename_axis(curve)


def nca_parameters(
    summary: pd.DataFrame,
    arms: pd.DataFrame,
    dose: str = "dose_mg_per_kg",
    route: str = "route",
    mol_wt: str = None,
    pair_on: list = ("compound_id",),
) -> pd.DataFrame:
    """Clearance, volumes and bioavailability from `nca()` output, in the ADMET reporting units.

    - IV curves: CL = Dose / AUCinf, Vss = CL · MRT, Vz = CL / lambda_z.
    - Other routes: F = dose-normalized AUCinf over the median dose-normalized AUCinf of
      the IV curves sharing the `pair_on` values. NaN when no IV curve matches.

    Args:
        summary (pd.DataFrame): `nca()` output.
        arms (pd.DataFrame): One row per curve, indexed like `summary`, carrying the dose,
            route and `pair_on` columns (and `mol_wt` when given).
        dose (str): Dose column, mg/kg. Defaults to "dose_mg_per_kg".
        route (str): Dosing route column. Defaults to "route".
        mol_wt (str): Molecular weight column (g/mol) when concentrations are nM; None when
            they are ng/mL. Defaults to None.
        pair_on (list): Columns an IV curve must share with a curve to be its F reference.
            Defaults to ("compound_id",).

    Returns:
        pd.DataFrame: NCA_PARAM_COLUMNS, indexed like `summary`.
    """
    arms = arms.reindex(summary.index)
    to_ng_ml = arms[mol_wt].astype(float) / 1000.0 if mol_wt else 1.0
    params = pd.DataFrame(index=summary.index, columns=NCA_PARAM_COLUMNS, dtype=float)
    for name in ("cmax", "auc_last", "auc_inf"):
        unit = "ng_ml" if name == "cmax" else "ng_h_ml"
        params[f"{name}_{unit}"] = summary[name] * to_ng_ml

    # mg/kg over ng·h/mL is 1000 L/h/kg
    iv = arms[route].isin(IV_ROUTES)
    dose_normalized = params["auc_inf_ng_h_ml"] / arms[dose].astype(float)
    cl_l_h_kg = (1000.0 / dose_normalized).where(iv)
    params["cl_ml_min_kg"] = cl_l_h_kg / CL_TO_L_PER_H
    params["vss_l_kg"] = cl_l_h_kg * summary["mrt"]
    params["vz_l_kg"] = cl_l_h_kg / summary["lambda_z"]

    keys = list(pair_on)
    reference = (
        arms.loc[iv, keys]
        .assign(iv_reference=dose_normalized[iv])
        .groupby(keys, as_index=False)["iv_reference"]
        .median()
    )
    paired = arms[keys].reset_index().merge(reference, on=keys, how="left").set_index(summary.index.name or "index")
    params["f_percent"] = (100.0 * dose_normalized / paired["iv_reference"]).where(~iv)
    return params
