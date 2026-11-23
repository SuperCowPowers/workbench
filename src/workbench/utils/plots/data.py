"""Data plots (matplotlib): how the rows of a keyed comparison overlap, where its one-sided rows concentrate,
which columns a quality check flagged, and a column's outliers against its spread.

Each takes an optional `ax` so a report can draw onto a page it is laying out; without one it
makes its own figure. Either way it returns the figure.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from workbench.utils.pandas_utils import DataQuality, KeyedComparison
from workbench.utils.plots.report import ACCENTS, BODY, HEADING_FONT, HEADING_WEIGHT, INK, MUTED, tidy

# A pair whose values disagree
MISMATCH_COLOR = ACCENTS[3]

# The most characters of a column name on a chart's axis
MAX_COLUMN_CHARS = 24

# The most groups a breakdown draws; the rest are summed into one bar
MAX_GROUPS = 8

# The height of one entry of the flagged-rows list, in inches
ITEM_PITCH = 0.36

# One colour per data quality check
CHECK_COLORS = {
    "missing": ACCENTS[1],
    "non_finite": ACCENTS[3],
    "unparseable": ACCENTS[2],
    "outliers": ACCENTS[4],
    "duplicate_key": ACCENTS[0],
    "duplicate_rows": MUTED,
}


def overlap_venn(
    result: KeyedComparison,
    left: str = "left",
    right: str = "right",
    title: str = "",
    locations: Optional[Dict[str, str]] = None,
    location_chars: int = 60,
    ax=None,
):
    """Two frames as overlapping circles: rows only in one, and paired rows that agree or differ.

    The circles are a fixed size, so a small region stays readable; the counts carry the scale.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        left (str): Name of the left frame.
        right (str): Name of the right frame.
        title (str): Left-aligned axes title.
        locations (Optional[Dict[str, str]]): Where a side's data lives, keyed by its name;
            shown small under that name. Leave it out on a chart drawn small.
        location_chars (int): Most characters of a location; a longer one keeps both ends.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(4.2, 2.8))[1]
    locations = locations or {}
    counts = result.summary()
    aside = counts["left_out_of_scope"] + counts["right_out_of_scope"]
    for x, color in ((-0.44, ACCENTS[1]), (0.44, ACCENTS[0])):
        ax.add_patch(Circle((x, 0), 1.0, fc=color, alpha=0.16, lw=0))
        ax.add_patch(Circle((x, 0), 1.0, fc="none", ec=color, lw=1.5))
    # The two names meet over the middle and each runs outward, so a long location has its own half of the page
    for x, side, name in ((-0.1, "right", left), (0.1, "left", right)):
        heading = {"fontname": HEADING_FONT, "fontweight": HEADING_WEIGHT}
        ax.text(x, 1.3, name, ha=side, va="bottom", fontsize=8, color=BODY, **heading)
        where = locations.get(name, "")
        if len(where) > location_chars:  # keep both ends: the root and the leaf are what identify it
            keep = (location_chars - 3) // 2
            where = f"{where[:keep]}...{where[-keep:]}"
        ax.text(x, 1.26, where, ha=side, va="top", fontsize=6.4, color=MUTED)

    def region(x: float, count: int, label: str) -> None:
        ax.text(
            x,
            0.06,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=INK,
            fontname=HEADING_FONT,
            fontweight=HEADING_WEIGHT,
        )
        ax.text(x, -0.04, label, ha="center", va="top", fontsize=7, color=BODY)

    region(-1.0, counts["left_only"], "only here")
    region(1.0, counts["right_only"], "only here")
    region(0, counts["agree"], "agree")
    # Under the circles: the paired rows that differ, by the counts the report states elsewhere, then rows set aside
    differing = [(counts["mismatched"], "mismatched"), (counts["repeated_differ"], "repeated keys differ")]
    below = [(", ".join(f"{count:,} {label}" for count, label in differing if count), MISMATCH_COLOR)]
    below += [(f"{aside:,} rows not expected, set aside" if aside else "", MUTED)]
    for row, (line, color) in enumerate((line, color) for line, color in below if line):
        ax.text(0, -1.1 - 0.2 * row, line, ha="center", va="top", fontsize=7, color=color)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.52, 1.56)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(
            title, loc="left", pad=7, fontsize=9.5, fontname=HEADING_FONT, fontweight=HEADING_WEIGHT, color=INK
        )
    return ax.figure


def one_sided_bars(
    result: KeyedComparison, by: str, left: str = "left", right: str = "right", title: str = "", ax=None
):
    """Where the rows held by one side only concentrate: a pair of bars per value of a column.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        by (str): The column to group the one-sided rows by. A side without it is left out.
        left (str): Name of the left frame.
        right (str): Name of the right frame.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If neither side's one-sided rows carry `by`.
    """
    sides = [(left, result.left_only, ACCENTS[1]), (right, result.right_only, ACCENTS[0])]
    sides = [(name, frame, color) for name, frame, color in sides if by in frame.columns]
    if not sides:
        raise ValueError(f"Neither side's one-sided rows have a column named {by}")
    counts = pd.DataFrame(
        {name: frame[by].astype("string").fillna("(none)").value_counts() for name, frame, _ in sides}
    ).fillna(0)
    if len(counts) > MAX_GROUPS:  # the largest groups, then everything else as one
        order = counts.sum(axis=1).sort_values(ascending=False, kind="stable").index
        rest = counts.loc[order[MAX_GROUPS - 1 :]].sum().rename("other")
        counts = pd.concat([counts.loc[order[: MAX_GROUPS - 1]].sort_index(), rest.to_frame().T])
    else:
        counts = counts.sort_index()

    ax = ax if ax is not None else plt.subplots(figsize=(4.2, 0.6 + 0.4 * len(counts)))[1]
    positions, height = np.arange(len(counts))[::-1], 0.8 / len(sides)
    for i, (name, _, color) in enumerate(sides):
        offsets = positions + 0.4 - height * (i + 0.5)
        ax.barh(offsets, counts[name], height=height * 0.86, color=color, lw=0, label=f"only in {name}")
        for offset, count in zip(offsets, counts[name]):
            if count:
                ax.text(count, offset, f" {int(count):,}", va="center", fontsize=7, color=BODY)
    ax.set_yticks(positions, [str(group) for group in counts.index])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.margins(x=0.12)
    ax.legend(frameon=False, fontsize=7.5, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.16))
    tidy(ax, title, grid="x")
    return ax.figure


def outlier_box(result: DataQuality, column: str, title: str = "", ax=None):
    """One column's spread as an upright box, with its fence and the values outside it marked.

    The box is drawn from the quartiles and whiskers `data_quality` kept, so the column itself
    is not needed. A column fenced on its log10 values gets a log axis; when an outlier is zero,
    the axis is linear near zero so the point has somewhere to sit.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.
        column (str): A column with an outliers finding.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.

    Raises:
        ValueError: If the column has no outliers finding.
    """
    held = result.boxes[result.boxes["column"] == column]
    if held.empty:
        raise ValueError(f"{column} has no outliers finding")
    box = held.iloc[0]
    listed = result.rows[(result.rows["check"] == "outliers") & (result.rows["column"] == column)]
    outliers = pd.to_numeric(listed["value"], errors="coerce").dropna().to_numpy()

    ax = ax if ax is not None else plt.subplots(figsize=(1.6, 3.2))[1]
    stats = {"med": box["median"], "q1": box["q1"], "q3": box["q3"]}
    stats.update(whislo=box["whisker_low"], whishi=box["whisker_high"], fliers=[])
    ax.bxp(
        [stats],
        orientation="vertical",
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": ACCENTS[1], "alpha": 0.18, "edgecolor": ACCENTS[1], "linewidth": 1.2},
        medianprops={"color": INK, "linewidth": 1.4},
        whiskerprops={"color": ACCENTS[1], "linewidth": 1.2},
        capprops={"color": ACCENTS[1], "linewidth": 1.2},
    )
    for fence in (box["fence_low"], box["fence_high"]):
        ax.axhline(fence, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(
        np.ones(len(outliers)), outliers, s=26, color=MISMATCH_COLOR, edgecolor="white", lw=0.8, zorder=3, clip_on=False
    )
    if box["log"] and (outliers <= 0).any():
        # Linear below a tenth of the smallest value inside the fence, with zero and the decades above it labelled
        threshold = box["whisker_low"] / 10
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_ylim(bottom=-threshold / 2)
        decades = np.arange(np.ceil(np.log10(threshold)) + 1, np.ceil(np.log10(ax.get_ylim()[1])))
        ax.set_yticks([0] + list(10.0**decades))
        ax.minorticks_off()
    elif box["log"]:
        ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    fence = f"fence {box['fence_low']:.3g} to {box['fence_high']:.3g}"
    ax.set_xlabel(f"{len(outliers):,} outside\n{fence}", fontsize=7).set_color(BODY)
    tidy(ax, title, grid="y")
    ax.spines["bottom"].set_visible(False)
    return ax.figure


def quality_counts(result: DataQuality) -> pd.DataFrame:
    """Flagged rows per column and check, worst column first.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.

    Returns:
        pd.DataFrame: Indexed by column, one column per check. A whole-row check such as
            `duplicate_rows` is indexed as "(whole row)".
    """
    counted = result.findings.dropna(subset=["count"])
    counted = counted.assign(column=counted["column"].fillna("(whole row)"), count=counted["count"].astype(int))
    table = counted.pivot_table(index="column", columns="check", values="count", aggfunc="sum", fill_value=0)
    return table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]


def quality_list(result: DataQuality, most: int = 8, title: str = "", ax=None):
    """Flagged rows per column as a list: the count, the column, and the checks behind the count.

    A count is coloured for a warning and plain for a minor finding, worst column first.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.
        most (int): Most columns listed. Defaults to 8; fewer when the axes is too short.
        title (str): Left-aligned axes title.
        ax: Axes to draw on. Defaults to a new figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    table = quality_counts(result)
    ax = ax if ax is not None else plt.subplots(figsize=(3.6, 0.3 + ITEM_PITCH * max(min(len(table), most), 1)))[1]
    warned = result.findings[result.findings["severity"] == "warn"]["column"].fillna("(whole row)")
    width, height = (
        ax.get_position().width * ax.figure.get_figwidth(),
        ax.get_position().height * ax.figure.get_figheight(),
    )
    pitch = ITEM_PITCH / height
    heading = {"fontname": HEADING_FONT, "fontweight": HEADING_WEIGHT}
    for row, (column, counts) in enumerate(table.head(min(most, int(1 / pitch))).iterrows()):
        y = 1 - row * pitch
        color = MISMATCH_COLOR if column in set(warned) else ACCENTS[1]
        ax.text(0.52 / width, y, f"{int(counts.sum()):,}", ha="right", va="top", fontsize=13, color=color, **heading)
        name = str(column) if len(str(column)) <= MAX_COLUMN_CHARS else f"{str(column)[: MAX_COLUMN_CHARS - 3]}..."
        ax.text(0.64 / width, y - 0.02 / height, name, va="top", fontsize=9, color=INK, **heading)
        checks = ", ".join(f"{check} {int(count):,}" for check, count in counts.items() if count)
        ax.text(0.64 / width, y - 0.18 / height, checks, va="top", fontsize=7.5, color=BODY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", pad=7, fontsize=9.5, color=INK, **heading)
    return ax.figure
