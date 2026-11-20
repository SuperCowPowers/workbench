"""Model performance plots (matplotlib): predicted against measured, residuals, and a
metric broken out by group.

Each takes an optional `ax` so a report can draw onto a page it is laying out; without
one it makes its own figure. Either way it returns the figure.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workbench.utils.plots.report import ACCENTS, BODY, MUTED, RULE, tidy

# Scatters above this many points are rasterized, which keeps a PDF small
RASTERIZE_ABOVE = 2000


def _axes(ax, figsize: tuple):
    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


def _pairs(y_true, y_pred, positive: bool = False) -> pd.DataFrame:
    """Aligned truth/prediction pairs with non-finite values (and, for log axes, non-positives) dropped."""
    pairs = pd.DataFrame({"true": np.asarray(y_true, dtype=float), "pred": np.asarray(y_pred, dtype=float)})
    keep = pd.Series(np.isfinite(pairs.to_numpy()).all(axis=1), index=pairs.index)
    if positive:
        keep &= (pairs > 0).all(axis=1)
    return pairs[keep]


def fold_bands(ax, lo: float, hi: float) -> None:
    """Fold-error guides on a log-log axis: the 1:1 line, a shaded 2x band, dashed 3x and
    dotted 10x lines.

    Args:
        ax: Axes to draw on; set to equal-aspect log scales spanning `lo` to `hi`.
        lo (float): Lower limit of both axes, > 0.
        hi (float): Upper limit of both axes.
    """
    x = np.array([lo, hi])
    ax.fill_between(x, x / 2, x * 2, color=RULE, alpha=0.6, lw=0, zorder=0)
    ax.plot(x, x, color=BODY, lw=1.0, zorder=1)
    for fold, style in ((3, "--"), (10, ":")):
        for k in (fold, 1 / fold):
            ax.plot(x, x * k, color=MUTED, lw=0.8, ls=style, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def pred_vs_actual(
    y_true,
    y_pred,
    groups=None,
    log: bool = False,
    unit: Optional[str] = None,
    title: str = "",
    color: str = ACCENTS[0],
    ax=None,
):
    """Predicted against measured, with the 1:1 line.

    Args:
        y_true: Measured values.
        y_pred: Predicted values, aligned with `y_true`.
        groups: Optional label per row; points are coloured by group, in `ACCENTS` order by
            group size, with groups beyond the palette pooled as "other".
        log (bool): Log axes with fold-error bands, for positive quantities. Non-positive
            pairs are left off. Defaults to False.
        unit (Optional[str]): Unit shown in the axis labels.
        title (str): Left-aligned axes title.
        color (str): Point colour when `groups` is not given.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = _axes(ax, (4.6, 4.6))
    pairs = _pairs(y_true, y_pred, positive=log)
    if log:
        lo = 10 ** np.floor(np.log10(pairs.to_numpy().min()))
        hi = 10 ** np.ceil(np.log10(pairs.to_numpy().max()))
        fold_bands(ax, lo, hi)
    else:
        lo, hi = pairs.to_numpy().min(), pairs.to_numpy().max()
        lo, hi = lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)
        ax.plot([lo, hi], [lo, hi], color=BODY, lw=1.0, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    style = dict(s=9, alpha=0.7, lw=0, zorder=2, rasterized=len(pairs) > RASTERIZE_ABOVE)
    if groups is None:
        ax.scatter(pairs["true"], pairs["pred"], color=color, **style)
    else:
        labels = pd.Series(np.asarray(groups)).reindex(pairs.index).fillna("unknown").astype(str)
        named = [label for label in labels.value_counts().index if label != "other"][: len(ACCENTS)]
        labels = labels.where(labels.isin(named), "other")
        for label, group_color in list(zip(named, ACCENTS)) + [("other", MUTED)]:
            hit = pairs[labels == label]
            if not hit.empty:
                ax.scatter(hit["true"], hit["pred"], color=group_color, label=label, **style)
        ax.legend(frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0.2, loc="upper left")

    suffix = f" ({unit})" if unit else ""
    ax.set_xlabel(f"Measured{suffix}", fontsize=8.5)
    ax.set_ylabel(f"Predicted{suffix}", fontsize=8.5)
    tidy(ax, title)
    return ax.figure


def residuals(
    y_true,
    y_pred,
    log: bool = False,
    unit: Optional[str] = None,
    bins: int = 30,
    title: str = "",
    color: str = ACCENTS[0],
    ax=None,
):
    """Distribution of the prediction error, with zero and the median marked.

    Args:
        y_true: Measured values.
        y_pred: Predicted values, aligned with `y_true`.
        log (bool): Plot log10(predicted / measured) rather than predicted - measured, for
            positive quantities. Defaults to False.
        unit (Optional[str]): Unit shown in the axis label (ignored when `log`).
        bins (int): Histogram bins. Defaults to 30.
        title (str): Left-aligned axes title.
        color (str): Bar colour.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = _axes(ax, (4.6, 3.2))
    pairs = _pairs(y_true, y_pred, positive=log)
    error = np.log10(pairs["pred"] / pairs["true"]) if log else pairs["pred"] - pairs["true"]
    ax.hist(error, bins=bins, color=color, alpha=0.85, lw=0)
    ax.axvline(0, color=BODY, lw=1.0)
    ax.axvline(error.median(), color=MUTED, lw=1.0, ls="--")
    label = "log10(predicted / measured)" if log else "Predicted - measured" + (f" ({unit})" if unit else "")
    ax.set_xlabel(label, fontsize=8.5)
    ax.set_ylabel("Rows", fontsize=8.5)
    tidy(ax, title, grid="y")
    return ax.figure


def breakout_bars(
    values: pd.Series,
    reference: Optional[float] = None,
    xlabel: str = "",
    title: str = "",
    color: str = ACCENTS[0],
    ax=None,
):
    """One metric per group as horizontal bars, each labelled with its value.

    Args:
        values (pd.Series): The metric, indexed by group, drawn top to bottom in the order given.
        reference (Optional[float]): A value to mark across every bar, e.g. the metric over all rows.
        xlabel (str): Axis label, normally the metric name.
        title (str): Left-aligned axes title.
        color (str): Bar colour.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = _axes(ax, (4.6, 0.6 + 0.28 * len(values)))
    positions = np.arange(len(values))[::-1]
    bars = ax.barh(positions, values.to_numpy(dtype=float), color=color, alpha=0.85, lw=0, height=0.62)
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=3, fontsize=7.5, color=BODY)
    if reference is not None:
        ax.axvline(reference, color=BODY, lw=1.0, ls="--")
    ax.set_yticks(positions, [str(label) for label in values.index])
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.margins(x=0.12)
    tidy(ax, title, grid="x")
    return ax.figure
