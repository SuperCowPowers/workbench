"""Report sections: each turns an analysis result into pages, plus the detail frames behind them.

A section's pages show counts and a few example rows; its frames hold every row, and the
Report publishes them so a reader can look the rest up.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from workbench.reports.report import Report, Section, Summary
from workbench.utils.metrics_utils import performance_breakout
from workbench.utils.pandas_utils import DataQuality, KeyedComparison
from workbench.utils.plots import data as data_charts
from workbench.utils.plots import performance as charts
from workbench.utils.plots import report as pages

# Rows of a detail frame shown on a page; the rest are in the published frame
EXAMPLE_ROWS = 8

# Columns whose outliers get a box plot on the data quality page
OUTLIER_BOXES = 3

# Rows shown behind each data quality warning
EXAMPLE_ROWS_PER_WARNING = 10

# Columns shown beside the key in a one-sided example table
EXTRA_COLUMNS = 3

# A table is not started, or continued, with fewer rows than this at the foot of a page
MIN_TABLE_ROWS = 3

SETS_DIFFER = "several rows share a key, and the two sides do not hold the same values"

# How a repeated key was judged, by `compare_keyed`'s `repeats`
REPEATS = {
    "exact": "agree when both sides hold the same rows: the same count with the same values",
    "values": "agree when every value is on both sides, however many rows carry it",
}

# Page geometry in figure units: the lowest a body may reach
# above the footer; a table's header; and one table row
BOTTOM, TABLE_HEAD, TABLE_ROW = 0.09, 0.03, 0.0172

METRIC_NOTES = (
    "R2 is the share of the target's variance the predictions explain. RMSE and MAE are in target units; "
    "RMSE weights large misses more. Spearman scores ranking alone."
)
LOG_METRIC_NOTES = (
    "R2, RMSE, MAE and Spearman are computed on log10 values, over the rows where measured and predicted are "
    "both positive. GMFE is the typical fold miss (1.5 means off by 50% in either direction), and within 2x "
    "is the share of predictions between half and double the measured value; 3x and 10x likewise."
)

# Fold bands reported for a positive quantity
FOLDS = ("2x", "3x", "10x")


def _num(value) -> str:
    """A cell as text: three significant figures for a float, blank for a missing value."""
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return ""
    return f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value)


def _is_number(values: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


def _clip(value, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _rows(df: pd.DataFrame, columns: List[str], limit: int = EXAMPLE_ROWS, width: int = 28) -> List[List[str]]:
    return [[_clip(_num(v), width) for v in row] for row in df[columns].head(limit).itertuples(index=False)]


def _rows_that_fit(height: float, most: int) -> int:
    """How many table rows fit in `height` of the page, under a header: at least 1, at most `most`."""
    return int(min(most, max(1, (height - TABLE_HEAD) // TABLE_ROW)))


def _where(report: Report, fig, y: float, frame: str, count: int, shown: Optional[int] = None) -> float:
    """A muted line saying how many rows were shown and where every row of the frame is.

    Says nothing about the location when the report is unpublished, and nothing at all when
    every row is on the page and there is no location to give.
    """
    location = report.location(frame)
    parts = [f"Showing the first {shown:,} of {count:,}." if shown is not None and shown < count else ""]
    parts += [f"All {count:,} rows: DFStore {location}" if location is not None and count else ""]
    note = " ".join(part for part in parts if part)
    return pages.para(fig, y, note, size=8.2, color=pages.MUTED) - 0.012 if note else y


def performance(
    df: pd.DataFrame,
    target: str,
    prediction: str = "prediction",
    by: Optional[str] = None,
    key: Union[str, List[str], None] = None,
    log: bool = False,
    unit: Optional[str] = None,
    name: str = "performance",
    about: Optional[str] = None,
) -> Section:
    """Model performance: metrics and charts, then where the model misses.

    Args:
        df (pd.DataFrame): Predictions, with the target and prediction columns.
        target (str): Measured value column.
        prediction (str): Predicted value column. Defaults to "prediction".
        by (Optional[str]): Column to break performance out on, e.g. "project".
        key (Union[str, List[str], None]): Column(s) identifying a row, shown beside the
            largest errors. Defaults to the index.
        log (bool): The target is a positive quantity: log axes and fold-error metrics.
            Defaults to False.
        unit (Optional[str]): Target unit for axis labels.
        name (str): Prefix for this section's frames. Defaults to "performance".
        about (Optional[str]): The model or endpoint scored, by the name the report uses for it.
            The report's notes on it are shown at the top of the page.

    Returns:
        Section: Two pages, and the frames `<name>_rows` (every row with its error, worst
            first) and `<name>_breakout` (the metrics table).
    """
    key_cols = [key] if isinstance(key, str) else list(key or [])
    breakout = performance_breakout(df, target, by=by, prediction_col=prediction, log=log)
    overall = breakout.iloc[0]

    if int(overall["support"]) == 0:
        raise ValueError(f"No row has both '{target}' and '{prediction}' to score")
    detail = df[list(dict.fromkeys(key_cols + ([by] if by else []) + [target, prediction]))].copy()
    if not key_cols:
        detail.insert(0, "index", df.index)
    if log:
        positive = (detail[target] > 0) & (detail[prediction] > 0)
        detail["log10_fold"] = np.log10((detail[prediction] / detail[target]).where(positive))
        error = "log10_fold"
    else:
        detail["error"] = detail[prediction] - detail[target]
        error = "error"
    detail = detail.sort_values(error, key=abs, ascending=False, na_position="last", kind="stable")
    rows_frame, breakout_frame = f"{name}_rows", f"{name}_breakout"

    def overview(report: Report):
        fig, y = report.new_page(f"Model performance: {target}", f"{int(overall['support']):,} rows scored.")
        y = report.note_block(fig, y, [about] if about else [])
        stats = [("R2", _num(overall["r2"])), ("RMSE", _num(overall["rmse"])), ("MAE", _num(overall["mae"]))]
        stats += [("Spearman", _num(overall["spearmanr"])), ("Rows", f"{int(overall['support']):,}")]
        if log:
            stats += [("GMFE", _num(overall["gmfe"]))]
            stats += [(f"Within {fold}", f"{overall[f'within_{fold}']:.0%}") for fold in FOLDS]
        y = pages.stat_tiles(fig, y, stats)
        y = pages.para(fig, y, LOG_METRIC_NOTES if log else METRIC_NOTES, size=8.6)

        scatter_title = f"{overall['within_2x']:.0%} of predictions within 2x" if log else f"R2 {overall['r2']:.2f}"
        groups = df[by] if by else None
        scatter = fig.add_axes([0.10, y - 0.40, 0.37, 0.33])
        charts.pred_vs_actual(
            df[target], df[prediction], groups=groups, log=log, unit=unit, title=scatter_title, ax=scatter
        )
        median = detail[error].median()
        spread = fig.add_axes([0.58, y - 0.40, 0.35, 0.33])
        charts.residuals(df[target], df[prediction], log=log, unit=unit, title=f"Median error {median:+.3g}", ax=spread)
        return fig

    def misses(report: Report):
        fig, y = report.new_page("Where the model misses")
        if by:
            metric = "gmfe" if log else "mae"
            within = [f"within_{fold}" for fold in FOLDS]
            columns = ["group", "support", "r2", "rmse"] + (["gmfe"] + within if log else ["mae"])
            shown = breakout.assign(**{c: breakout[c].map("{:.0%}".format) for c in within}) if log else breakout
            y = pages.sec(fig, y, f"By {by}")
            y = pages.table(fig, y, columns, _rows(shown, columns, limit=len(shown)), hl_row=0)
            groups = breakout.iloc[1:].set_index("group")[metric]
            bars = fig.add_axes([0.24, y - 0.05 - 0.026 * len(groups), 0.50, 0.026 * len(groups) + 0.02])
            worst = groups.idxmax() if len(groups) else ""
            charts.breakout_bars(
                groups, overall[metric], metric.upper(), f"{worst} has the highest {metric.upper()}", ax=bars
            )
            y -= 0.13 + 0.026 * len(groups)
            y = _where(report, fig, y, breakout_frame, len(breakout))
        y = pages.sec(fig, y, "Largest errors")
        columns = list(detail.columns)
        shown = min(len(detail), _rows_that_fit(y - BOTTOM - 0.03, most=10))
        y = pages.table(fig, y, columns, _rows(detail, columns, limit=shown))
        _where(report, fig, y - 0.008, rows_frame, len(detail), shown=shown)
        return fig

    if log:
        headline = f"{overall['within_2x']:.0%} of {target} predictions are within 2x of the measured value."
        stats = [("GMFE", _num(overall["gmfe"])), ("Within 3x", f"{overall['within_3x']:.0%}")]
    else:
        headline = f"Predictions explain {overall['r2']:.0%} of the variance in {target}."
        stats = [("RMSE", _num(overall["rmse"])), ("MAE", _num(overall["mae"]))]
    if by and len(breakout) > 2:
        metric = "gmfe" if log else "mae"
        headline += f" {breakout.iloc[1:].set_index('group')[metric].idxmax()} is the weakest {by}."
    stats += [("Spearman", _num(overall["spearmanr"])), ("Rows", f"{int(overall['support']):,}")]
    card = Summary(
        "Model performance",
        headline,
        stats,
        lambda ax: charts.pred_vs_actual(df[target], df[prediction], groups=df[by] if by else None, log=log, ax=ax),
        hero=(f"{overall['within_2x']:.0%}", "within 2x") if log else (f"{overall['r2']:.2f}", "R2"),
    )
    frames = {rows_frame: detail, breakout_frame: breakout}
    return Section(pages=[overview, misses], frames=frames, summary=card, about=[about] if about else [])


def data_quality(
    result: DataQuality,
    title: str = "Data quality",
    name: str = "quality",
    about: Optional[str] = None,
) -> Section:
    """Data quality: the findings, then example rows for each warning.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.
        title (str): Page headline. Defaults to "Data quality".
        name (str): Prefix for this section's frames. Defaults to "quality".
        about (Optional[str]): The artifact the frame came from, by the name the report uses
            for it. The report's notes on that artifact are shown at the top of the page.

    Returns:
        Section: One or two pages, and the frames `<name>_findings` and `<name>_rows`
            (every offending cell).
    """
    findings, rows = result.findings, result.rows
    key_cols = [c for c in rows.columns if c not in ("check", "column", "value")]
    warnings = int((findings["severity"] == "warn").sum())
    minor = int((findings["severity"] == "info").sum())
    about = [about] if about else []
    flagged = int(findings["count"].sum())
    per_column = data_charts.quality_counts(result).sum(axis=1)
    findings_frame, rows_frame = f"{name}_findings", f"{name}_rows"
    counted = [(warnings, "warning", "warnings"), (minor, "minor finding", "minor findings")]
    verdict = " and ".join(f"{count} {one if count == 1 else many}" for count, one, many in counted) + "."
    verdict = verdict if warnings or minor else "No problems found."

    def summary(report: Report):
        fig, y = report.new_page(title, verdict)
        y = report.note_block(fig, y, about)
        y = pages.stat_tiles(fig, y, [("Warnings", warnings), ("Minor", minor), ("Cells flagged", f"{flagged:,}")])
        if findings.empty:
            return fig
        if len(per_column):
            height = 0.028 * min(len(per_column), 10) + 0.03
            bars = fig.add_axes([0.22, y - height - 0.04, 0.62, height])
            worst = per_column.idxmax()
            data_charts.quality_bars(result, title=f"{worst} has the most flagged rows", ax=bars)
            y -= height + 0.115
        for _, box in result.boxes.head(OUTLIER_BOXES).iterrows():
            count = int(
                findings[(findings["check"] == "outliers") & (findings["column"] == box["column"])]["count"].iloc[0]
            )
            heading = f"{box['column']}: {count:,} outside the fence" + (" (log scale)" if box["log"] else "")
            data_charts.outlier_box(
                result, box["column"], title=heading, ax=fig.add_axes([0.22, y - 0.075, 0.62, 0.042])
            )
            y -= 0.115
        shown = findings.drop(columns="count").fillna("")
        shown = shown.assign(detail=shown["detail"].map(lambda d: _clip(d, 50)))
        y = pages.sec(fig, y, "Findings, worst first")
        columns = ["check", "column", "severity", "value", "detail"]
        limit = _rows_that_fit(y - BOTTOM - 0.03, most=30)
        y = pages.table(fig, y, columns, _rows(shown, columns, limit=limit, width=60))
        _where(report, fig, y - 0.008, findings_frame, len(findings), shown=min(limit, len(findings)))
        return fig

    def examples(report: Report):
        each = EXAMPLE_ROWS_PER_WARNING
        fig, y = report.new_page(f"{title}: rows to look at", f"Up to {each} rows behind each warning.")
        warned = findings[findings["severity"] == "warn"][["check", "column"]]
        behind = rows.merge(warned, on=["check", "column"])
        picks = behind.groupby(["check", "column"], sort=False, dropna=False).head(each)
        columns = ["check", "column"] + key_cols + ["value"]
        shown = _rows_that_fit(y - BOTTOM - 0.03, most=len(picks))
        y = pages.table(fig, y, columns, _rows(picks.fillna(""), columns, limit=shown, width=90))
        # Say rows were held back only when a warning has more than are on the page
        _where(report, fig, y - 0.008, rows_frame, len(rows), shown=shown if shown < len(behind) else None)
        return fig

    headline = verdict
    if len(per_column):
        headline += f" {per_column.idxmax()} has the most flagged rows ({int(per_column.max()):,})."
    card = Summary(
        title,
        headline,
        [("Warnings", warnings), ("Minor", minor), ("Cells flagged", f"{flagged:,}")],
        (lambda ax: data_charts.quality_bars(result, most=5, ax=ax)) if len(per_column) else None,
        hero=(str(warnings), "warning" if warnings == 1 else "warnings"),
        needs_action=bool(warnings),
    )
    has_examples = warnings and not rows.empty
    return Section(
        pages=[summary] + ([examples] if has_examples else []),
        frames={findings_frame: findings, rows_frame: rows},
        summary=card,
        about=about,
    )


def comparison(
    result: KeyedComparison,
    left: str = "left",
    right: str = "right",
    title: Optional[str] = None,
    name: str = "comparison",
    max_rows: int = 10,
    locations: Optional[Dict[str, str]] = None,
    by: Optional[str] = None,
) -> Section:
    """Data comparison: what is fine and what needs a look, then the rows that need it.

    Fine is every pair that agrees, plus rows set aside by `scope` as not expected on the
    other side. Needing a look is mismatched pairs, rows on one side only, and repeated keys
    whose two sides hold different values.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        left (str): What to call the left frame in the document.
        right (str): What to call the right frame.
        title (Optional[str]): Page headline. Defaults to "<left> against <right>".
        name (str): Prefix for this section's frames. Defaults to "comparison".
        max_rows (int): Most rows shown per bucket in the rows to look at. Defaults to 10;
            every row is in the bucket's frame, whose location is printed under the table.
            The tables run onto further pages when they do not fit on one.
        locations (Optional[Dict[str, str]]): Where a side's data lives (a network drive, an S3
            prefix), keyed by its name. Shown under that name on the overlap chart, and in
            full with how the rows were compared.
        by (Optional[str]): A column to break the one-sided rows down by (a year, a route, a
            project), drawn beside the overlap chart. Rows of a side that lacks the column
            are left out of it.

    Returns:
        Section: The overview page, the rows to look at, and a frame per bucket: `<name>_mismatched` (one row per
            disagreeing value), `_left_only`, `_right_only`, `_repeated_differ`,
            `_left_out_of_scope`, `_right_out_of_scope`.
    """
    title = title or f"{left} against {right}"
    on, pairs, counts = result.on, result.pairs, result.summary()
    values = result.values
    numeric = [v for v in values if all(_is_number(pairs[f"{v}_{side}"]) for side in ("left", "right"))]
    disagreements = []
    for value in values:
        off = pairs[~pairs[f"{value}_agree"]]
        if value in numeric:  # largest relative difference first
            a, b = (off[f"{value}_{side}"].to_numpy(dtype=float, na_value=np.nan) for side in ("left", "right"))
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = np.abs(a - b) / np.abs(b)
            off = off.iloc[np.argsort(-np.nan_to_num(gap, nan=-1.0), kind="stable")]
        sides = {left: off[f"{value}_left"].astype(str), right: off[f"{value}_right"].astype(str)}
        disagreements.append(off[on].assign(column=value, **sides))
    mismatched = pd.concat(disagreements, ignore_index=True) if disagreements else pd.DataFrame(columns=on)
    differ = result.repeated_differ.drop(columns="values_match")
    differ = differ.sort_values(["left_rows", "right_rows"], ascending=False, kind="stable")
    renamed = {"left_rows": f"{left} rows", "right_rows": f"{right} rows"}
    renamed.update({f"{v}_left": f"{v} in {left}" for v in values})
    renamed.update({f"{v}_right": f"{v} in {right}" for v in values})
    differ = differ.rename(columns=renamed)

    # What needs a look: frame suffix, what the document calls it, its rows, its size, what it means
    look = [
        (
            "mismatched",
            "mismatched",
            mismatched,
            f"{counts['mismatched']:,} pairs",
            "one row on each side, values disagree",
        ),
        ("left_only", f"{left} only", result.left_only, None, f"key not in {right}"),
        ("right_only", f"{right} only", result.right_only, None, f"key not in {left}"),
        ("repeated_differ", "repeated key, values differ", differ, f"{len(differ):,} keys", SETS_DIFFER),
    ]
    # What was set aside as not expected: the side it came from, the side that never holds it, its rows
    aside = [
        ("left_out_of_scope", left, right, result.left_out_of_scope),
        ("right_out_of_scope", right, left, result.right_out_of_scope),
    ]
    frames = {f"{name}_{suffix}": frame for suffix, _, frame, _, _ in look}
    frames.update({f"{name}_{suffix}": frame for suffix, _, _, frame in aside})
    verdict = f"{counts['agree']:,} of {counts['pairs']:,} paired rows agree."
    # How the rows were compared, stated so a reader knows what "agree" means here
    if result.rtol and result.atol:  # the two allowances add
        tolerance = f"within {result.atol:g} plus {result.rtol:.3g} of the {right} value"
    elif result.rtol:
        tolerance = f"within {result.rtol:.3g} of the {right} value"
    else:
        tolerance = f"within {result.atol:g}" if result.atol else "exactly"
    method = [[f"{side} is at", where] for side, where in (locations or {}).items() if side in (left, right)]
    method += [["matched on", ", ".join(on)]]
    method += [["compared", f"{', '.join(values)}; numbers agree {tolerance}"]] if values else []
    method += [[f"limited to {col}", ", ".join(str(v) for v in held)] for col, held in result.scope.items()]
    method += [["rows sharing a key", REPEATS[result.repeats]]]
    phrases = [  # a count, then what one of them and several of them are called
        (counts["mismatched"], "mismatched pair", "mismatched pairs"),
        (counts["left_only"], f"row only in {left}", f"rows only in {left}"),
        (counts["right_only"], f"row only in {right}", f"rows only in {right}"),
        (counts["repeated_differ"], "repeated key whose values differ", "repeated keys whose values differ"),
    ]
    open_items = [f"{count:,} {one if count == 1 else many}" for count, one, many in phrases if count]

    def overview(report: Report):
        fig, y = report.new_page(title, verdict)
        y = report.note_block(fig, y, [left, right])
        stats = [("Agree", counts["agree"]), ("Mismatched", counts["mismatched"])]
        stats += [(f"{left} only", counts["left_only"]), (f"{right} only", counts["right_only"])]
        stats += [("Repeated keys differ", counts["repeated_differ"])]
        y = pages.stat_tiles(fig, y, [(k, f"{v:,}") for k, v in stats])
        y = pages.table(fig, y, ["how rows were compared", ""], method, aligns=["left", "left"]) - 0.006
        if by is None:
            venn = fig.add_axes([pages.L, y - 0.2, pages.R - pages.L, 0.2])
            data_charts.overlap_venn(result, left, right, locations=locations, ax=venn)
        else:  # the breakdown of one-sided rows on the left, the overlap on the right
            bars = fig.add_axes([pages.L + 0.07, y - 0.185, 0.36, 0.145])
            data_charts.one_sided_bars(result, by, left, right, title=f"Rows on one side only, by {by}", ax=bars)
            venn = fig.add_axes([0.54, y - 0.2, pages.R - 0.54, 0.2])
            data_charts.overlap_venn(result, left, right, locations=locations, location_chars=36, ax=venn)
        y -= 0.225

        y = pages.sec(fig, y, "Fine")
        fine = [["agree", f"{counts['agree']:,} pairs", "the two sides hold the same values"]]
        for _, source, other, frame in aside:
            if not frame.empty:
                meaning = f"{source} rows with a {' or '.join(result.scope)} {other} never holds"
                fine.append([f"not expected in {other}", f"{len(frame):,} rows", meaning])
        y = pages.table(fig, y, ["bucket", "size", "meaning"], fine, aligns=["left", "right", "left"]) - 0.012
        y = pages.sec(fig, y, "Needs a look")
        rows = [[label, shown or f"{len(frame):,} rows", meaning] for suffix, label, frame, shown, meaning in look]
        y = pages.table(fig, y, ["bucket", "size", "meaning"], rows, aligns=["left", "right", "left"])

        return fig

    def examples(report: Report):
        heading = f"{title}: rows to look at"
        fig, y = report.new_page(heading)
        figs = [fig]

        def flow(suffix: str, label: str, meaning: str, columns: list, rows: list, total: int, sample: bool):
            """One table, continued onto as many pages as its rows need.

            `sample` is True for a table listing the frame's own rows, which says how many
            of them it shows, and False for one summarising them.
            """
            nonlocal fig, y
            color = pages.INK if sample else pages.MUTED
            start = 0
            while start < len(rows):
                room = int((y - BOTTOM - TABLE_HEAD - 0.07) // TABLE_ROW)
                if room < min(MIN_TABLE_ROWS, len(rows) - start):
                    fig, y = report.new_page(heading, "Continued.")
                    figs.append(fig)
                    continue
                marked = label
                y = pages.sec(fig, y, f"{marked}: {meaning}" if start == 0 else f"{marked}, continued", color)
                y = pages.table(fig, y, columns, rows[start : start + room])
                start += room
            y = _where(report, fig, y - 0.004, f"{name}_{suffix}", total, shown=len(rows) if sample else None) - 0.012

        for suffix, label, frame, _, meaning in look:
            if frame.empty:
                continue
            if suffix == "mismatched":
                parsed = frame[[left, right]].apply(pd.to_numeric, errors="coerce")
                shown = {s: parsed[s].map(_num).where(parsed[s].notna(), frame[s]) for s in (left, right)}
                frame, columns = frame.assign(**shown), list(frame.columns)
            elif suffix == "repeated_differ":
                columns = list(frame.columns)
            else:  # the key, then the first few other columns
                columns = on + [c for c in frame.columns if c not in on][:EXTRA_COLUMNS]
            rows = _rows(frame, columns, limit=max_rows, width=22)
            flow(suffix, label, meaning, columns, rows, len(frame), True)
        for suffix, source, other, frame in aside:
            if frame.empty:
                continue
            reasons = frame.groupby("scope_reason").size().sort_values(ascending=False, kind="stable")
            rows = [[reason, f"{count:,}"] for reason, count in reasons.items()]
            meaning = f"{source} rows it holds no counterpart for"
            flow(suffix, f"Not expected in {other}", meaning, ["reason", "rows"], rows, len(frame), False)
        return figs

    headline = verdict + (f" Needs a look: {', '.join(open_items)}." if open_items else " Nothing needs a look.")
    stats = [("Agree", f"{counts['agree']:,}"), ("Mismatched", f"{counts['mismatched']:,}")]
    stats += [(f"{left} only", f"{counts['left_only']:,}"), (f"{right} only", f"{counts['right_only']:,}")]
    card = Summary(
        title,
        headline,
        stats,
        lambda ax: data_charts.overlap_venn(result, left, right, ax=ax),
        hero=(f"{counts['agree']:,}", f"of {counts['pairs']:,} paired rows agree"),
        needs_action=bool(open_items),
    )
    has_examples = bool(open_items) or any(not frame.empty for *_, frame in aside)
    pages_ = [overview] + ([examples] if has_examples else [])
    return Section(pages=pages_, frames=frames, summary=card, about=[left, right])
