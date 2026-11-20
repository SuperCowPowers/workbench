"""Data plots (matplotlib): where the rows of a keyed comparison went, how paired values agree,
and which columns a quality check flagged.

Each takes an optional `ax` so a report can draw onto a page it is laying out; without one it
makes its own figure. Either way it returns the figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workbench.utils.pandas_utils import DataQuality, KeyedComparison
from workbench.utils.plots.report import ACCENTS, BODY, MUTED, RULE, tidy

# One colour per fate of a compared row, kept across charts and pages
FATE_COLORS = {
    "agree": ACCENTS[0],
    "mismatched": ACCENTS[3],
    "ambiguous": ACCENTS[2],
    "one side only": ACCENTS[1],
    "out of scope": RULE,
}

# One colour per data quality check
CHECK_COLORS = {
    "missing": ACCENTS[1],
    "non_finite": ACCENTS[3],
    "unparseable": ACCENTS[2],
    "outliers": ACCENTS[4],
    "duplicate_key": ACCENTS[0],
    "duplicate_rows": MUTED,
}


def comparison_fates(result: KeyedComparison) -> pd.DataFrame:
    """Rows of each side of a comparison by what became of them.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.

    Returns:
        pd.DataFrame: Indexed by fate (agree, mismatched, ambiguous, one side only, out of
            scope) with `left` and `right` row counts. Each column sums to its frame's rows.
    """
    paired = {"agree": int(result.pairs["agree"].sum()), "mismatched": len(result.mismatched)}
    return pd.DataFrame(
        {
            "left": {
                **paired,
                "ambiguous": int(result.ambiguous["n_left"].sum()),
                "one side only": len(result.left_only),
                "out of scope": len(result.left_out_of_scope),
            },
            "right": {
                **paired,
                "ambiguous": int(result.ambiguous["n_right"].sum()),
                "one side only": len(result.right_only),
                "out of scope": len(result.right_out_of_scope),
            },
        }
    )


def fate_bars(result: KeyedComparison, left: str = "left", right: str = "right", title: str = "", ax=None):
    """Where every row of a comparison went: one stacked bar per side.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        left (str): Label for the left frame's bar.
        right (str): Label for the right frame's bar.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(6.4, 1.6))[1]
    fates = comparison_fates(result)
    for position, side in ((1, "left"), (0, "right")):
        start = 0
        for fate, count in fates[side].items():
            ax.barh(position, count, left=start, color=FATE_COLORS[fate], lw=0, height=0.58, label=fate)
            if count >= 0.06 * fates[side].sum():
                dark = fate == "out of scope"
                ax.text(start + count / 2, position, f"{count:,}", ha="center", va="center", fontsize=7.5).set_color(
                    BODY if dark else "white"
                )
            start += count
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(), frameon=False, fontsize=7.5, ncols=5, loc="upper left", bbox_to_anchor=(0, -0.3)
    )
    ax.set_yticks([1, 0], [left, right])
    ax.set_ylim(-0.5, 1.5)
    tidy(ax, title, grid="x")
    return ax.figure


def value_agreement(
    result: KeyedComparison, value: str, left: str = "left", right: str = "right", title: str = "", ax=None
):
    """One compared value, left against right, for every uniquely keyed pair.

    Agreeing pairs sit on the 1:1 line; the pairs that disagree are drawn on top in a second
    colour. Axes are logarithmic when every value is positive and they span over two decades.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        value (str): A numeric column the comparison scored.
        left (str): Axis label for the left frame.
        right (str): Axis label for the right frame.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(4.6, 4.6))[1]
    pairs = pd.DataFrame(
        {
            "left": result.pairs[f"{value}_left"].to_numpy(dtype=float, na_value=np.nan),
            "right": result.pairs[f"{value}_right"].to_numpy(dtype=float, na_value=np.nan),
            "agree": result.pairs[f"{value}_agree"].to_numpy(dtype=bool),
        }
    )
    pairs = pairs[np.isfinite(pairs[["left", "right"]]).all(axis=1)]
    values = pairs[["left", "right"]].to_numpy()
    if len(values):
        lo, hi = values.min(), values.max()
        if lo > 0 and hi / lo > 100:
            lo, hi = 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            lo, hi = lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)
        ax.plot([lo, hi], [lo, hi], color=BODY, lw=1.0, zorder=1)
        if hi > lo:
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    style = dict(s=9, lw=0, rasterized=len(pairs) > 2000)
    ax.scatter(
        pairs.loc[pairs.agree, "right"], pairs.loc[pairs.agree, "left"], color=MUTED, alpha=0.6, zorder=2, **style
    )
    off = pairs[~pairs.agree]
    ax.scatter(off["right"], off["left"], color=FATE_COLORS["mismatched"], alpha=0.9, zorder=3, **style)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{value} in {right}", fontsize=8.5)
    ax.set_ylabel(f"{value} in {left}", fontsize=8.5)
    tidy(ax, title)
    return ax.figure


def quality_bars(result: DataQuality, most: int = 10, title: str = "", ax=None):
    """Flagged rows per column, stacked by check, worst column first.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.
        most (int): Most columns drawn. Defaults to 10.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    counted = result.findings.dropna(subset=["count"])
    counted = counted.assign(column=counted["column"].fillna("(whole row)"))
    table = counted.pivot_table(index="column", columns="check", values="count", aggfunc="sum", fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index[:most]]
    ax = ax if ax is not None else plt.subplots(figsize=(6.4, 0.7 + 0.3 * max(len(table), 1)))[1]
    positions = np.arange(len(table))[::-1]
    start = np.zeros(len(table))
    for check in table.columns:
        color = CHECK_COLORS.get(check, BODY)
        ax.barh(positions, table[check], left=start, color=color, alpha=0.9, lw=0, height=0.62, label=check)
        start += table[check].to_numpy(dtype=float)
    for position, total in zip(positions, start):
        ax.text(total, position, f" {int(total):,}", va="center", fontsize=7.5, color=BODY)
    if len(table):
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_yticks(positions, [str(c) for c in table.index])
    ax.set_xlabel("Rows flagged", fontsize=8.5)
    ax.margins(x=0.1)
    tidy(ax, title, grid="x")
    return ax.figure
