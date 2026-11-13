"""Pharmacokinetic plots: the interactive Bateman profile (Plotly, knobs are sliders) and
measured concentration-time curves with their NCA fit (matplotlib, for reports)."""

import numpy as np
import pandas as pd

from workbench.utils.pk_utils import bateman_concentration, bateman_tmax

# Absorption rate constants to put on the slider, in 1/h. Spans flip-flop kinetics
# (ka below ke, absorption rate-limiting) through fast absorption.
KA_RANGE = (0.1, 5.0)
KA_STEPS = 25


def bateman(
    volume: float,
    clearance: float,
    f_percent: float = 100.0,
    dose: float = 100.0,
    ka_range: tuple = KA_RANGE,
    ka_steps: int = KA_STEPS,
    duration: float = None,
    n_points: int = 300,
    title: str = None,
):
    """Plasma concentration-time profile with absorption rate on a slider.

    The single oral dose, one-compartment model with first-order absorption and
    first-order elimination -- the Bateman function. Also called a PK profile,
    a concentration-time curve, or an exposure profile.

    Volume, clearance, and bioavailability come from the compound; the elimination rate
    `ke = clearance / volume` falls out of the first two. Absorption rate `ka` is the one
    parameter an early ADMET package rarely pins down, so it is the slider: drag it to see
    how much of the exposure profile is absorption-limited. AUC is annotated because it
    does *not* move with ka -- only the shape does.

    Units are whatever you feed it, as long as they agree: volume in L and clearance in
    L/h give ke in 1/h, and a dose in mg then reads as mg/L on the y-axis.

    Args:
        volume (float): Volume of distribution, > 0.
        clearance (float): Clearance, > 0. Same volume unit as `volume`, per hour.
        f_percent (float): Oral bioavailability as a percentage. Defaults to 100.0.
        dose (float): Administered dose. Defaults to 100.0.
        ka_range (tuple): Low/high absorption rate constant (1/h) for the slider.
        ka_steps (int): Slider positions, log-spaced across `ka_range`. Defaults to 25.
        duration (float, optional): Hours to plot. Defaults to five half-lives of whichever
            process is slower -- elimination, or absorption at the low end of `ka_range`.
        n_points (int): Time samples per curve. Defaults to 300.
        title (str, optional): Plot title. A sensible default is built when None.

    Returns:
        plotly.graph_objects.Figure: The figure. In the REPL `fig.show()` opens a browser
            tab; the slider needs that (or a notebook), so it has no matplotlib equivalent.
    """
    import plotly.graph_objects as go

    if volume <= 0 or clearance <= 0:
        raise ValueError(f"volume and clearance must be > 0, got volume={volume}, clearance={clearance}")
    if not 0 < f_percent <= 100:
        raise ValueError(f"f_percent is a percentage in (0, 100], got {f_percent}")
    if dose <= 0:
        raise ValueError(f"dose must be > 0, got {dose}")
    lo, hi = ka_range
    if not 0 < lo < hi:
        raise ValueError(f"ka_range must be an increasing pair above zero, got {ka_range}")

    ke = clearance / volume
    half_life = np.log(2) / ke
    f_dose = dose * f_percent / 100.0
    auc = f_dose / clearance  # independent of ka -- the point of the annotation

    # Whichever of absorption/elimination is slower sets the terminal slope, and the
    # window is shared across slider positions -- so size it by the slowest ka on offer,
    # or the tail truncates in the flip-flop regime and AUC stops looking constant.
    rate_limiting = min(ke, lo)
    duration = duration if duration is not None else 5 * np.log(2) / rate_limiting
    t = np.linspace(0, duration, n_points)
    ka_values = np.geomspace(lo, hi, ka_steps)
    active = int(np.argmin(np.abs(ka_values - 1.0)))

    # Two traces per slider position: the curve, and a marker at its peak.
    fig = go.Figure()
    for i, ka in enumerate(ka_values):
        conc = bateman_concentration(t, f_dose, volume, ke, ka)
        peak_t = bateman_tmax(ke, ka)
        peak_c = float(bateman_concentration(np.array([peak_t]), f_dose, volume, ke, ka)[0])
        fig.add_trace(
            go.Scatter(
                x=t,
                y=conc,
                mode="lines",
                line=dict(width=3),
                name="concentration",
                hovertemplate="t=%{x:.2f} h<br>C=%{y:.3f}<extra></extra>",
                visible=(i == active),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak_t],
                y=[peak_c],
                mode="markers+text",
                marker=dict(size=11, symbol="circle-open", line=dict(width=3)),
                text=[f"  Cmax {peak_c:.3g}"],
                textposition="middle right",
                hovertemplate=f"Tmax={peak_t:.2f} h<br>Cmax={peak_c:.3g}<extra></extra>",
                showlegend=False,
                visible=(i == active),
            )
        )

    def _readout(ka):
        peak_t = bateman_tmax(ke, ka)
        regime = "absorption-limited (flip-flop)" if ka < ke else "elimination-limited"
        return (
            f"<b>ka</b> {ka:.2f} /h &nbsp; <b>ke</b> {ke:.3f} /h &nbsp; <b>t½</b> {half_life:.2f} h<br>"
            f"<b>Tmax</b> {peak_t:.2f} h &nbsp; <b>AUC</b> {auc:.3g} (fixed)<br>"
            f"<i>{regime}</i>"
        )

    def _annotation(ka):
        return dict(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            align="left",
            showarrow=False,
            borderwidth=1,
            borderpad=8,
            font=dict(size=13),
            text=_readout(ka),
        )

    steps = []
    for i, ka in enumerate(ka_values):
        visible = [False] * (2 * ka_steps)
        visible[2 * i] = visible[2 * i + 1] = True
        steps.append(
            dict(
                method="update",
                label=f"{ka:.2f}",
                args=[{"visible": visible}, {"annotations": [_annotation(ka)]}],
            )
        )

    fig.add_annotation(**_annotation(ka_values[active]))

    if title is None:
        title = f"Bateman profile — V={volume:g}, CL={clearance:g}, F={f_percent:g}%, dose={dose:g}"
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title=dict(text="Time (h)", font=dict(size=14)), rangemode="tozero"),
        yaxis=dict(title=dict(text="Concentration", font=dict(size=14)), rangemode="tozero"),
        showlegend=False,
        height=620,
        margin=dict(t=90, b=90),
        sliders=[
            dict(
                active=active,
                currentvalue=dict(prefix="ka = ", suffix=" /h", font=dict(size=14)),
                pad=dict(t=50),
                steps=steps,
            )
        ],
    )
    return fig


def nca_curves(
    curves: pd.DataFrame,
    curve_ids: list,
    summary: pd.DataFrame = None,
    labels: dict = None,
    ncols: int = 4,
    curve: str = "curve_id",
    time: str = "time_h",
    conc: str = "conc",
    blq: str = "blq",
):
    """Measured concentration-time curves as small multiples on a log axis, one panel per curve.

    Quantifiable samples are joined; BLQ samples are hollow markers at half the curve's
    lowest quantifiable concentration. With `summary`, each panel overlays the terminal
    log-linear fit across the samples it used and prints t½, extrapolated AUC % and the
    fit's adjusted R².

    Args:
        curves (pd.DataFrame): One row per sample, as passed to `pk_utils.nca`.
        curve_ids (list): Curves to draw, in panel order.
        summary (pd.DataFrame, optional): `pk_utils.nca` output for the fit overlay.
        labels (dict, optional): Panel title per curve id. Defaults to the curve id.
        ncols (int): Panels per row. Defaults to 4.
        curve (str): Column identifying the curve. Defaults to "curve_id".
        time (str): Sample time column. Defaults to "time_h".
        conc (str): Concentration column. Defaults to "conc".
        blq (str): Boolean below-quantitation column, or None for none. Defaults to "blq".

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt

    from workbench.utils.plots.report import ACCENTS, BODY, MUTED

    curve_ids = list(curve_ids)
    ncols = min(ncols, len(curve_ids))
    nrows = -(-len(curve_ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.95 * ncols, 1.75 * nrows), squeeze=False)
    for ax, curve_id in zip(axes.flat, curve_ids):
        group = curves[curves[curve] == curve_id].sort_values(time)
        t, c = group[time].to_numpy(float), group[conc].to_numpy(float, na_value=np.nan)
        flagged = group[blq].fillna(False).to_numpy(bool) if blq else np.zeros(len(group), dtype=bool)
        measured = flagged | ~np.isnan(c)
        t, c, below = t[measured], c[measured], (flagged | ~(c > 0))[measured]
        tq, cq = t[~below], c[~below]

        ax.set_yscale("log")
        ax.plot(tq, cq, "-o", color=ACCENTS[1], markersize=3, linewidth=1)
        if below.any() and len(cq):
            ax.plot(t[below], np.full(below.sum(), cq.min() / 2), "o", mfc="none", mec=MUTED, markersize=3)

        fitted = summary is not None and curve_id in summary.index and pd.notna(summary.at[curve_id, "lambda_z"])
        if fitted:
            row = summary.loc[curve_id]
            lambda_z, n = row["lambda_z"], int(row["lambda_z_n"])
            # a least-squares line passes through the centroid of the samples it was fit to
            intercept = np.mean(np.log(cq[-n:]) + lambda_z * tq[-n:])
            ends = tq[-n:][[0, -1]]
            ax.plot(ends, np.exp(intercept - lambda_z * ends), color=ACCENTS[3], linewidth=1.3)
            note = f"t½ {row['half_life']:.3g} h\nextrap {row['extrap_percent']:.0f}%\nR² {row['lambda_z_r2']:.2f}"
            ax.text(0.96, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=BODY)
            # headroom on the log axis keeps the note clear of the samples
            low, high = ax.get_ylim()
            ax.set_ylim(low, high * (high / low) ** 0.6)
        ax.set_title(labels.get(curve_id, curve_id) if labels else curve_id, fontsize=8, loc="left")
        ax.tick_params(which="both", labelsize=7)
    for ax in axes.flat[len(curve_ids) :]:
        ax.set_visible(False)
    fig.supxlabel("time (h)", fontsize=8)
    fig.supylabel("concentration", fontsize=8)
    fig.tight_layout()
    return fig
