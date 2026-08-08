"""Plots for hyperparameter searches."""

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("workbench")


def _as_dict(cell) -> dict:
    """Read a JSON cell; the HPO frames carry these as strings."""
    if isinstance(cell, dict):
        return cell
    return json.loads(cell) if isinstance(cell, str) and cell else {}


def _trial_counts(runs: pd.DataFrame) -> dict:
    """Derive the search's outcome counts when the results dict doesn't carry them.

    An incomplete trial with a value was pruned; one that never scored failed.
    """
    done = runs["completed"].astype(bool) if "completed" in runs else pd.Series(True, index=runs.index)
    scored = runs["value"].notna()
    return {
        "attempted": len(runs),
        "completed": int(done.sum()),
        "pruned": int((~done & scored).sum()),
        "failed": int((~done & ~scored).sum()),
    }


def _fmt(value) -> str:
    """A knob value at readable precision -- `0.000158`, not `0.00015804170058407083`.

    Whole numbers keep their digits: `.3g` would render a width of 2400 as `2.4e+03`.
    """
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return str(value)
    return str(int(number)) if float(number).is_integer() else f"{number:.3g}"


def _trial_value(config: dict, runs: pd.DataFrame, knob_frame: pd.DataFrame) -> Optional[float]:
    """The objective of the trial that ran this exact config, on the trials' own basis."""
    if not config:
        return None
    match = pd.Series(True, index=knob_frame.index)
    for knob, value in config.items():
        if knob in knob_frame.columns:
            match &= knob_frame[knob].astype(str) == str(value)
    hits = runs.loc[match[match].index, "value"]
    return float(hits.iloc[0]) if len(hits) else None


# Samples per segment of a curved line. Also the marker stride: the coordinates land on
# every Nth point of the curve, so one plot call can carry both the line and its markers.
_PER_SEGMENT = 32


def _curved_path(y):
    """Smooth a polyline through its coordinates, flat where it crosses each axis.

    Each segment is a smoothstep (`3t^2 - 2t^3`), whose derivative is zero at both ends, so
    a line settles onto its value at every axis instead of cutting through at an angle. The
    coordinates are the control points and the values there stay exact.
    """
    y = np.asarray(y, dtype=float)
    xs, ys = [], []
    for i in range(len(y) - 1):
        t = np.linspace(0, 1, _PER_SEGMENT, endpoint=False)
        xs.append(i + t)
        ys.append(y[i] + (y[i + 1] - y[i]) * (t * t * (3 - 2 * t)))
    xs.append(np.array([len(y) - 1.0]))
    ys.append(np.array([y[-1]]))
    return np.concatenate(xs), np.concatenate(ys)


# Overlaid lines accumulate ink as 1-(1-alpha)^k, so holding the crowd's density steady
# as trials grow means alpha ~ 1/n. Anchored where 0.35 reads well, then clamped: the cap
# keeps small runs from going garish, the floor keeps one line from vanishing in a big one.
_ALPHA_ANCHOR = (60, 0.35)  # (trials, opacity that reads well at that count)
_ALPHA_LIMITS = (0.10, 0.35)


def _line_alpha(n_trials: int) -> float:
    """Per-line opacity for a run of this many trials."""
    ref_trials, ref_alpha = _ALPHA_ANCHOR
    return float(np.clip(ref_alpha * ref_trials / max(n_trials, 1), *_ALPHA_LIMITS))


def _attach_hover(
    fig, ax, artists: list, paths, records: list, metric: str, base_alpha: float, radius_px: float = 20.0
):
    """Show a trial's config when the cursor is near its line, nearest line winning.

    Distances are measured in display pixels rather than data units, so "nearest" matches
    what the eye sees regardless of the axis scales. Needs an interactive backend -- under
    a headless one no motion events fire and this is simply inert.
    """
    paths = np.asarray(paths, dtype=float)
    note = ax.annotate(
        "",
        xy=(0, 0),
        textcoords="offset points",
        fontsize=10,
        zorder=20,
        visible=False,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe8", ec="#888888", alpha=0.95),
    )
    active = [None]

    def nearest(event):
        """Index of the line under the cursor, or None when nothing is close enough."""
        if event.inaxes is not ax:
            return None
        screen = ax.transData.transform(paths.reshape(-1, 2)).reshape(paths.shape)
        gaps = np.hypot(screen[..., 0] - event.x, screen[..., 1] - event.y)
        with np.errstate(invalid="ignore"):
            per_line = np.where(np.all(np.isnan(gaps), axis=1), np.inf, np.nanmin(gaps, axis=1))
        winner = int(np.argmin(per_line))
        return winner if per_line[winner] <= radius_px else None

    def show(winner, event):
        record = records[winner]
        knobs = "\n".join(f"  {k} = {_fmt(v)}" for k, v in record["config"].items())
        note.set_text(f"trial {record['number']} — {metric} {record['value']:.4f}\n{knobs}")
        note.xy = (event.xdata, event.ydata)
        # Open toward the middle, so the tooltip stays inside the axes near a border.
        rightward = event.xdata < sum(ax.get_xlim()) / 2
        upward = event.ydata < sum(ax.get_ylim()) / 2
        note.set_position((14 if rightward else -14, 14 if upward else -14))
        note.set_ha("left" if rightward else "right")
        note.set_va("bottom" if upward else "top")
        note.set_visible(True)

    def on_move(event):
        winner = nearest(event)
        if winner == active[0]:
            return
        if active[0] is not None:
            artists[active[0]].set(linewidth=2.2, alpha=base_alpha, zorder=2)
        if winner is None:
            note.set_visible(False)
        else:
            artists[winner].set(linewidth=4.0, alpha=1.0, zorder=9)
            show(winner, event)
        active[0] = winner
        fig.canvas.draw_idle()

    # Held on the figure so the callback outlives this call.
    fig._hpo_hover = (fig.canvas.mpl_connect("motion_notify_event", on_move), note)


def _build_axis(knob: str, space_row, observed: pd.Series) -> dict:
    """One parallel-coordinates axis, typed from the search space's `dist`.

    The declared `low`/`high` win over the observed range: an axis scaled to what the
    search *could* have explored is what makes a knob pinned against its own bound
    visible. Falls back to the observed range only when the space doesn't declare one.
    """
    spec = _as_dict(space_row["spec"]) if space_row is not None else {}
    dist = space_row["dist"] if space_row is not None else None
    numeric = pd.to_numeric(observed, errors="coerce")

    if dist == "choice" or (dist is None and numeric.isna().any()):
        options = [str(o) for o in spec.get("options") or sorted({str(v) for v in observed.dropna()})]
        return {"knob": knob, "categorical": True, "options": options, "label": knob}

    low, high = spec.get("low"), spec.get("high")
    valid = numeric.dropna()
    if low is None or high is None:
        low = float(valid.min()) if len(valid) else 0.0
        high = float(valid.max()) if len(valid) else 1.0
    low, high = float(low), float(high)
    use_log = bool(spec.get("log")) and low > 0 and high > 0
    lo, hi = (np.log10(low), np.log10(high)) if use_log else (low, high)
    return {
        "knob": knob,
        "categorical": False,
        "log": use_log,
        "lo": lo,
        "hi": hi if hi > lo else lo + 1.0,
        "raw_lo": low,
        "raw_hi": high,
        "label": f"log10({knob})" if use_log else knob,
    }


def _position(axis: dict, value) -> float:
    """Place a raw knob value on its axis, normalized to [0, 1] and clipped.

    Clipping keeps a value outside the declared bounds -- a config the user set by hand,
    a default outside its own searched range -- drawn in-frame rather than off the axes.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if axis["categorical"]:
        options = axis["options"]
        if str(value) not in options:
            return np.nan
        return options.index(str(value)) / max(len(options) - 1, 1)
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    number = np.log10(number) if axis["log"] and number > 0 else number
    return float(np.clip((number - axis["lo"]) / (axis["hi"] - axis["lo"]), 0.0, 1.0))


def hpo_parallel_coordinates(
    model: Any,
    completed_only: bool = False,
    use_curves: bool = True,
    figsize: tuple = (16, 8),
    cmap_name: str = "RdBu_r",
    title: str = None,
) -> Optional[Any]:
    """Parallel-coordinates view of a model's hyperparameter search.

    One vertical axis per knob, one line per trial, colored by the objective. Shows which
    regions of the space the good trials cluster in and lets you trace a single config
    across every axis.

    Color is centered on the baseline -- the user's own untuned hyperparameters, scored on
    the same basis -- so hue answers "did this trial beat my defaults" rather than where it
    ranks within the run. The scale reaches as far as the published config beat the
    baseline, so the hues resolve the margin actually won; worse trials saturate. The
    baseline and the published config are drawn as reference lines; a search plot without
    the baseline can't show whether the search achieved anything.

    Args:
        model: A searched Workbench model (`hpo_results()` returns None if it wasn't).
        completed_only (bool): Plot only the trials that produced an objective. Trials that
            died are otherwise drawn alongside them, since where the search looked is part of
            the picture. Defaults to False.
        use_curves (bool): Draw each line as a smooth curve through its coordinates, flat
            where it crosses each axis, rather than straight segments. Values at the axes
            are unchanged either way. Defaults to True.
        figsize (tuple): Figure size. Defaults to (16, 8).
        cmap_name (str): Divergent colormap. The default "RdBu_r" puts the favorable hue
            on better-than-baseline for a minimized objective (MAE).
        title (str, optional): Plot title. A sensible default is built when None.

    Returns:
        matplotlib.figure.Figure: The figure, or None when the model was not searched.
            The caller shows or saves it: `fig.show()`, or
            `fig.savefig(path, dpi=150, bbox_inches="tight")`.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    results = model.hpo_results()
    if results is None or results.get("trials") is None:
        log.warning(f"No HPO results for {getattr(model, 'name', model)} -- was it hyperparameter-searched?")
        return None

    trials = results["trials"]
    metric = results.get("metric") or "objective"

    # The baseline row carries the caller's own hyperparameters on the search basis.
    baseline_row = trials[trials.get("kind").eq("baseline")] if "kind" in trials else trials.iloc[0:0]
    baseline_value = float(baseline_row["value"].iloc[0]) if len(baseline_row) else results.get("baseline_value")

    runs = trials[trials["kind"].eq("trial")] if "kind" in trials else trials
    counts = _trial_counts(runs)
    if completed_only and "completed" in runs:
        runs = runs[runs["completed"].astype(bool)]
    runs = runs.dropna(subset=["value"])  # a failed trial never scored
    if runs.empty:
        log.warning("No scored trials to plot.")
        return None

    # Expand each trial's knobs. The space describes the framework's full set and a search
    # may have used a subset, so drive off the knobs the trials actually carry.
    knob_frame = pd.DataFrame([_as_dict(h) for h in runs["hyperparameters"]], index=runs.index)
    space = model.hpo_search_space()
    space_rows = {row["knob"]: row for _, row in space.iterrows()} if space is not None else {}
    if space is None:
        log.warning("No HPO search space for this framework; scaling axes to the observed trial values.")

    # Most important knob on the left: only *adjacent* axes show their relationship, so the
    # order decides what the chart can reveal. Falls back to the space's own order.
    importance = model.hpo_importance()
    ordered = list(importance["knob"]) if importance is not None else list(space_rows)
    knobs = [k for k in ordered if k in knob_frame.columns]
    by_importance = importance is not None and bool(knobs)
    knobs = [k for k in (knobs or knob_frame.columns) if knob_frame[k].notna().any()]
    if not knobs:
        log.warning("Trials carry no knob values to plot.")
        return None
    axes_def = [_build_axis(k, space_rows.get(k), knob_frame[k]) for k in knobs]

    # Color is a *margin over baseline*, not a raw score: the question a reader has is
    # "did this trial beat my defaults", which a raw scale cannot answer.
    values = runs["value"].astype(float)
    center = float(baseline_value) if baseline_value is not None else float(values.median())
    deltas = values - center

    # Symmetric, so equal improvement and regression read equally far, and scaled by the best
    # margin won: every trial runs to term, so a hopeless config lands arbitrarily far above
    # the baseline and would otherwise flatten every real difference to white. Past either
    # end the color saturates, which is the right reading for those. The 10% keeps the
    # published marker off the colorbar's edge.
    reach = -float(deltas.min())
    if reach <= 0:
        # Nothing beat the baseline, so span the worse side instead -- otherwise the scale
        # collapses to a point and every trial reads as baseline.
        reach = max(float(deltas.max()), 1e-9)
    span = reach * 1.1
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    mappable = ScalarMappable(norm=norm, cmap=cmap_name)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = list(range(len(axes_def)))

    # The axes themselves, behind everything: they anchor where a value is read off.
    for xi in x:
        ax.axvline(xi, color="#d8d8d8", lw=1.0, zorder=0)

    def _path(y):
        return _curved_path(y) if use_curves else (x, y)

    # Every trial drawn the same: the point is where blue and red sit along each axis, and
    # splitting the population by outcome would break that read.
    # Worst first, so the better trials still land on top of the crowd.
    artists, paths, records = [], [], []
    line_alpha = _line_alpha(len(values))
    for idx in deltas.sort_values(ascending=False).index:
        y = [_position(axis, knob_frame.at[idx, axis["knob"]]) for axis in axes_def]
        px, py = _path(y)
        (artist,) = ax.plot(px, py, color=mappable.to_rgba(deltas[idx]), lw=2.2, alpha=line_alpha, zorder=2)
        artists.append(artist)
        paths.append(np.column_stack([px, py]))
        records.append(
            {
                "number": runs.at[idx, "number"] if "number" in runs else idx,
                "value": float(values[idx]),
                "margin": float(deltas[idx]),
                "config": {axis["knob"]: knob_frame.at[idx, axis["knob"]] for axis in axes_def},
            }
        )

    # Reference lines: the baseline and the published config, opaque and on top.
    def _reference(config: dict, color: str, linestyle: str, label: str, lw: float = 3.0):
        y = [_position(axis, config.get(axis["knob"])) for axis in axes_def]
        if all(np.isnan(v) for v in y):
            return
        # The curve passes through each coordinate, so markevery lands the markers there.
        ax.plot(
            *_path(y),
            linestyle=linestyle,
            marker="o",
            markevery=_PER_SEGMENT if use_curves else 1,
            color=color,
            lw=lw,
            markersize=9,
            label=label,
            zorder=10,
        )
        for xi, axis, yi in zip(x, axes_def, y):
            value = config.get(axis["knob"])
            if value is None or np.isnan(yi):
                continue
            # Drop the label below a vertex sitting at the top, where the axis-max label is,
            # and tuck the end labels inward so they don't hang over the spines.
            above = yi < 0.9
            ax.text(
                xi,
                yi + (0.03 if above else -0.03),
                _fmt(value),
                ha="left" if xi == x[0] else "right" if xi == x[-1] else "center",
                va="bottom" if above else "top",
                fontsize=10,
                color=color,
                zorder=11,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
            )

    if len(baseline_row):
        _reference(_as_dict(baseline_row["hyperparameters"].iloc[0]), "#333333", "--", "baseline (default config)")
    if results.get("best_config"):
        _reference(results["best_config"], "#1b7837", "-", "published config", lw=4.2)

    ax.set_xticks(x)
    ax.set_xticklabels([axis["label"] for axis in axes_def], fontsize=11)
    if by_importance:
        ax.set_xlabel("importance: high → low", fontsize=10, color="#777777", labelpad=10)
    ax.set_yticks([])
    ax.set_ylim(-0.08, 1.08)
    for xi, axis in enumerate(axes_def):
        bottom, top = (
            (axis["options"][0], axis["options"][-1])
            if axis["categorical"]
            else (_fmt(axis["raw_lo"]), _fmt(axis["raw_hi"]))
        )
        ax.text(xi, -0.03, bottom, ha="center", va="top", fontsize=11, color="#333333")
        ax.text(xi, 1.03, top, ha="center", va="bottom", fontsize=11, color="#333333")

    bar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(f"{metric} vs baseline (negative is better)", fontsize=12)
    # Both rules live on the colorbar, which is scaled in *margin over baseline* -- so the
    # baseline sits at zero and the published config at its own margin. Drawing raw objective
    # values here would put both lines off the end of the scale.
    bar.ax.axhline(0.0, color="black", linestyle="--", lw=1.5)
    # Looked up across *all* scored rows, baseline included: a baseline win is a supported
    # outcome, and that row is not in `runs`.
    scored = trials.dropna(subset=["value"])
    published_value = _trial_value(
        results.get("best_config"),
        scored,
        pd.DataFrame([_as_dict(h) for h in scored["hyperparameters"]], index=scored.index),
    )
    if published_value is not None:
        rule = bar.ax.axhline(published_value - center, color="#00d451", lw=3.5)
        # A white halo so it reads against the dark end of the colormap.
        rule.set_path_effects([path_effects.withStroke(linewidth=6.0, foreground="white")])
    # Below the axes: an in-plot legend covers the top-of-axis value labels. The counts ride
    # along as the legend's title so they stay laid out with it.
    tally = (
        f"{counts['attempted']} trials — {counts['completed']} complete, "
        f"{counts['pruned']} pruned, {counts['failed']} failed"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=11,
        frameon=False,
        title=tally,
        title_fontsize=11,
    )
    if title is None:
        title = f"{getattr(model, 'name', 'model')} — HPO trials colored by {metric} vs baseline"
    ax.set_title(title, fontsize=14, pad=18)
    _attach_hover(fig, ax, artists, paths, records, metric, line_alpha)

    # Solve the layout once and freeze it. Left live, the solver re-runs on every redraw, so
    # a hover tooltip near an edge would shift the whole plot out from under the cursor.
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    return fig
