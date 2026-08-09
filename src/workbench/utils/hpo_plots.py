"""Plots for hyperparameter searches."""

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("workbench")

# Divergent, with the favorable hue on better-than-baseline for a minimized objective (MAE).
_CMAP = "RdBu_r"

# A stopped trial with nothing to estimate its shortfall from. Grey rather than a guess: its
# objective covers fewer ensemble members, so as measured it belongs on no shared scale.
_STOPPED_GREY = "#c4c4c4"

# Samples per segment of a curved line. Also the marker stride: the coordinates land on
# every Nth point of the curve, so one plot call can carry both the line and its markers.
_PER_SEGMENT = 32

# Overlaid lines accumulate ink as 1-(1-alpha)^k, so holding the crowd's density steady
# as trials grow means alpha ~ 1/n. Anchored where 0.35 reads well, then clamped: the cap
# keeps small runs from going garish, the floor keeps one line from vanishing in a big one.
_ALPHA_ANCHOR = (60, 0.35)  # (trials, opacity that reads well at that count)
_ALPHA_LIMITS = (0.10, 0.35)


def _as_dict(cell) -> dict:
    """Read a JSON cell; the HPO frames carry these as strings."""
    if isinstance(cell, dict):
        return cell
    return json.loads(cell) if isinstance(cell, str) and cell else {}


def _fmt(value) -> str:
    """A knob value at readable precision -- `0.000158`, not `0.00015804170058407083`.

    Whole numbers keep their digits: `.3g` would render a width of 2400 as `2.4e+03`.
    """
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return str(value)
    return str(int(number)) if float(number).is_integer() else f"{number:.3g}"


def _line_alpha(n_trials: int) -> float:
    """Per-line opacity for a run of this many trials."""
    ref_trials, ref_alpha = _ALPHA_ANCHOR
    return float(np.clip(ref_alpha * ref_trials / max(n_trials, 1), *_ALPHA_LIMITS))


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


def _attach_hover(fig, ax, artists: list, paths, records: list, metric: str, base_alpha: float, radius_px=20.0):
    """Show a trial's config when the cursor is near its line, nearest line winning.

    Distances are measured in display pixels rather than data units, so "nearest" matches
    what the eye sees regardless of the axis scales. Needs an interactive backend -- under
    a headless one no motion events fire and this is simply inert.
    """
    paths = np.asarray(paths, dtype=float)
    note = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
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
        # A stopped trial is shown at its estimate, so the tooltip carries both that and the
        # partial ensemble it was actually measured on.
        head = f"trial {record['number']} — {metric} {record['value']:.4f}"
        if record["estimate"] is not None:
            head = (
                f"trial {record['number']} — {metric} {record['estimate']:.4f} estimated\n"
                f"stopped at member {record['stopped_at']}, measured {record['value']:.4f}"
            )
        elif record["stopped_at"] is not None:
            head += f"\nstopped at member {record['stopped_at']}"
        knobs = "\n".join(f"  {k} = {_fmt(v)}" for k, v in record["config"].items())
        note.set_text(f"{head}\n{knobs}")
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


def _knob_axes(model: Any, runs: pd.DataFrame) -> tuple:
    """The axes to draw, most important knob first, plus the knob values behind them.

    Only *adjacent* axes show their relationship, so the order decides what the chart can
    reveal; `hpo_importance()` supplies it, the space's own order is the fallback. The space
    covers the framework's full set and a search may use a subset, so the knobs come from
    what the trials carry.
    """
    knob_frame = pd.DataFrame([_as_dict(h) for h in runs["hyperparameters"]], index=runs.index)
    space = model.hpo_search_space()
    if space is None:
        log.warning("No HPO search space for this framework; scaling axes to the observed trial values.")
    space_rows = {row["knob"]: row for _, row in space.iterrows()} if space is not None else {}

    importance = model.hpo_importance()
    ordered = list(importance["knob"]) if importance is not None else list(space_rows)
    knobs = [k for k in ordered if k in knob_frame.columns]
    by_importance = importance is not None and bool(knobs)
    knobs = [k for k in (knobs or knob_frame.columns) if knob_frame[k].notna().any()]
    return [_build_axis(k, space_rows.get(k), knob_frame[k]) for k in knobs], knob_frame, by_importance


def _objective_scale(values: pd.Series, completed: pd.Series, center: float):
    """A diverging scale in the metric's own units, centred on the baseline.

    Symmetric, and reaching only as far as the best margin won: a hopeless config lands
    arbitrarily far above the baseline and would otherwise flatten every real difference to
    white. Past either end the colour saturates, which is the right reading for those. Only
    completed trials set the reach, so an estimate placed here later cannot stretch it.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    scored = values[completed] if completed.any() else values
    reach = center - float(scored.min())
    if reach <= 0:
        # Nothing beat the baseline, so span the worse side instead -- otherwise the scale
        # collapses to a point and every trial reads as baseline.
        reach = max(float(scored.max()) - center, 1e-9)
    span = reach * 1.1
    norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
    return ScalarMappable(norm=norm, cmap=_CMAP)


def _fold_offsets(runs: pd.DataFrame, completed: pd.Series, full_step) -> dict:
    """How far the objective moves between fold *k* and the whole ensemble, per fold.

    Measured on the completed trials, the only ones carrying both ends of a trajectory: the
    stopped ones lost at a rung, so their early folds are not a fair sample of one. The median
    assumes fold difficulty shifts every config alike -- a wide spread in ``traj[K] - traj[k]``
    means it does not.
    """
    # `pd.isna`, not falsiness: NaN is truthy, and a frame can reach here with no usable step.
    if "trajectory" not in runs or pd.isna(full_step):
        return {}
    gaps = {}
    for idx in runs.index[completed.to_numpy()]:
        trajectory = {int(fold): float(v) for fold, v in _as_dict(runs.at[idx, "trajectory"]).items()}
        final = trajectory.get(int(full_step))
        if final is None:
            continue
        for fold, value in trajectory.items():
            if fold < full_step:
                gaps.setdefault(fold, []).append(final - value)
    return {fold: float(np.median(deltas)) for fold, deltas in gaps.items()}


def _draw_reference(ax, axes_def: list, config: dict, *, color: str, linestyle: str, label: str, lw=3.0) -> None:
    """A named config drawn opaque over the crowd, with its value labelled at each axis."""
    y = [_position(axis, config.get(axis["knob"])) for axis in axes_def]
    if all(np.isnan(v) for v in y):
        return
    # The curve passes through each coordinate, so markevery lands the markers there.
    ax.plot(
        *_curved_path(y),
        linestyle=linestyle,
        marker="o",
        markevery=_PER_SEGMENT,
        color=color,
        lw=lw,
        markersize=9,
        label=label,
        zorder=10,
    )
    for xi, (axis, yi) in enumerate(zip(axes_def, y)):
        value = config.get(axis["knob"])
        if value is None or np.isnan(yi):
            continue
        # Drop the label below a vertex sitting at the top, where the axis-max label is, and
        # tuck the end labels inward so they don't hang over the spines.
        above = yi < 0.9
        ax.text(
            xi,
            yi + (0.03 if above else -0.03),
            _fmt(value),
            ha="left" if xi == 0 else "right" if xi == len(axes_def) - 1 else "center",
            va="bottom" if above else "top",
            fontsize=10,
            color=color,
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
        )


def _label_axes(ax, axes_def: list, by_importance: bool) -> None:
    """Knob names below the axes, and each axis's own range at its ends."""
    ax.set_xticks(range(len(axes_def)))
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


def _draw_colorbar(fig, ax, mappable, metric: str, center: float, published_value) -> None:
    """The objective scale, with the baseline and the published config ruled onto it."""
    import matplotlib.patheffects as path_effects

    bar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(f"{metric} (lower is better)", fontsize=12)
    # Both rules are raw objective values, which is what the colorbar is scaled in.
    bar.ax.axhline(center, color="black", linestyle="--", lw=1.5)
    if published_value is not None:
        # The winner's own score, straight from the record. Matching by config would return
        # the first row holding those knobs, wrong whenever a config was evaluated twice.
        rule = bar.ax.axhline(published_value, color="#00d451", lw=3.5)
        # A white halo so it reads against the dark end of the colormap.
        rule.set_path_effects([path_effects.withStroke(linewidth=6.0, foreground="white")])


def _draw_legend(ax, counts: dict, estimated: bool) -> None:
    """The reference lines, below the axes, and the caveat when any hue is an estimate.

    Below rather than in-plot, where it would cover the top-of-axis value labels. The counts
    ride along as the legend's title so they stay laid out with it.
    """
    from matplotlib.lines import Line2D

    handles = ax.get_legend_handles_labels()[0]
    if estimated:
        handles.append(Line2D([], [], color="none", label="* pruned trials are estimates"))
    tally = (
        f"{counts['attempted']} trials — {counts['completed']} complete, "
        f"{counts['pruned']} pruned, {counts['failed']} failed"
    )
    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(handles) or 1,
        fontsize=11,
        frameon=False,
        title=tally,
        title_fontsize=11,
    )
    if estimated:
        legend.get_texts()[-1].set_style("italic")


def hpo_parallel_coordinates(model: Any, figsize: tuple = (16, 8), title: str = None) -> Optional[Any]:
    """Parallel-coordinates view of a model's hyperparameter search.

    One vertical axis per knob, one line per trial. Shows which regions of the space the good
    trials cluster in and lets you trace a single config across every axis.

    Hue is the objective on a scale diverging at the baseline -- the user's own untuned
    hyperparameters, scored on the same folds -- so it answers "did this beat my defaults"
    while the ticks read as the metric. A stopped trial was measured on fewer ensemble
    members, which on some datasets reads *better* than a full run, so it is carried onto the
    same scale by what the completed trajectories say the missing members cost. Those hues
    are estimates and the legend says so; with no trajectory to fit on, a stopped trial stays
    grey. The baseline and published config are reference lines: a search plot without the
    baseline can't show whether the search achieved anything.

    Args:
        model: A searched Workbench model (`hpo_results()` returns None if it wasn't).
        figsize (tuple): Figure size. Defaults to (16, 8).
        title (str, optional): Plot title. A sensible default is built when None.

    Returns:
        matplotlib.figure.Figure: The figure, or None when the model was not searched.
            The caller shows or saves it: `fig.show()`, or
            `fig.savefig(path, dpi=150, bbox_inches="tight")`.
    """
    import matplotlib.pyplot as plt

    results = model.hpo_results()
    if results is None or results.get("trials") is None:
        log.warning(f"No HPO results for {getattr(model, 'name', model)} -- was it hyperparameter-searched?")
        return None

    trials = results["trials"]
    metric = results.get("metric") or "objective"

    # The baseline row carries the caller's own hyperparameters on the search basis; the
    # record is the fallback when the frame carries no such row.
    baseline_row = trials[trials["kind"].eq("baseline")]
    baseline_value = float(baseline_row["value"].iloc[0]) if len(baseline_row) else results.get("search_baseline_value")

    # Counts cover every trial the budget paid for, the baseline included; `runs` drops it
    # only because it is drawn as a reference line rather than as one of the crowd.
    counts = results["trial_counts"]
    runs = trials[trials["kind"].eq("trial")].dropna(subset=["value"])  # a failed trial never scored
    if runs.empty:
        log.warning("No scored trials to plot.")
        return None

    axes_def, knob_frame, by_importance = _knob_axes(model, runs)
    if not axes_def:
        log.warning("Trials carry no knob values to plot.")
        return None

    values = runs["value"].astype(float)
    # `pd.notna`, not `is not None`: a failed baseline comes back from CSV as NaN, and a NaN
    # centre would put the whole scale off rather than falling back to the median.
    center = float(baseline_value) if pd.notna(baseline_value) else float(values.median())
    completed = runs["completed"].astype(bool)
    # `step` postdates the earliest searches; without it nothing can be estimated.
    steps = pd.to_numeric(runs["step"], errors="coerce") if "step" in runs else pd.Series(np.nan, index=runs.index)
    stopped_at = steps.where(~completed)

    mappable = _objective_scale(values, completed, center)
    # A stopped trial's objective covers fewer members than a completed one's, so it does not
    # belong on this scale as measured. The completed trials' trajectories say what the
    # missing members are worth, which carries it onto the scale as an estimate. Without a
    # trajectory to fit that on -- an artifact from before the column existed -- it stays off.
    full_step = steps[completed].max() if completed.any() else None
    offsets = _fold_offsets(runs, completed, full_step)
    estimates = {
        idx: values[idx] + offsets[stopped_at[idx]]
        for idx in runs.index[(~completed).to_numpy()]
        if stopped_at[idx] in offsets
    }
    # What each trial is drawn at: its own objective, or the estimate standing in for one.
    # Both the hue and the draw order come off this, so they cannot disagree.
    shown = values.copy()
    shown.update(pd.Series(estimates))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for xi in range(len(axes_def)):
        ax.axvline(xi, color="#d8d8d8", lw=1.0, zorder=0)  # anchors where a value is read off

    # Worst first, so the better a trial scored the closer to the top of the pile it lands --
    # by what it is drawn at, so an estimate ranks alongside a measured objective rather than
    # behind it. A grey has no place in that order at all and sits under the whole crowd.
    artists, paths, records = [], [], []
    line_alpha = _line_alpha(len(values))
    for idx in shown.sort_values(ascending=False).index:
        y = [_position(axis, knob_frame.at[idx, axis["knob"]]) for axis in axes_def]
        px, py = _curved_path(y)
        rung = stopped_at[idx]
        done = bool(completed[idx])
        scaled = done or idx in estimates
        color = mappable.to_rgba(shown[idx]) if scaled else _STOPPED_GREY
        (artist,) = ax.plot(px, py, color=color, lw=2.2, alpha=line_alpha, zorder=2 if scaled else 1)
        artists.append(artist)
        paths.append(np.column_stack([px, py]))
        records.append(
            {
                "number": runs.at[idx, "number"] if "number" in runs else idx,
                "value": float(values[idx]),
                "estimate": None if done else estimates.get(idx),
                "stopped_at": None if done or pd.isna(rung) else int(rung),
                "config": {axis["knob"]: knob_frame.at[idx, axis["knob"]] for axis in axes_def},
            }
        )

    if len(baseline_row):
        config = _as_dict(baseline_row["hyperparameters"].iloc[0])
        _draw_reference(ax, axes_def, config, color="#333333", linestyle="--", label="baseline (default config)")
    if results.get("best_config"):
        _draw_reference(
            ax, axes_def, results["best_config"], color="#1b7837", linestyle="-", label="published config", lw=4.2
        )

    _label_axes(ax, axes_def, by_importance)
    _draw_colorbar(fig, ax, mappable, metric, center, results.get("search_best_value"))
    _draw_legend(ax, counts, estimated=bool(estimates))
    ax.set_title(title or f"{getattr(model, 'name', 'model')} — HPO trials colored by {metric}", fontsize=14, pad=18)
    _attach_hover(fig, ax, artists, paths, records, metric, line_alpha)

    # Solve the layout once and freeze it. Left live, the solver re-runs on every redraw, so
    # a hover tooltip near an edge would shift the whole plot out from under the cursor.
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    return fig
