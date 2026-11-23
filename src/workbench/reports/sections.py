"""Report sections: each turns an analysis result into pages, plus the detail frames behind them.

A section's pages show counts and a few example rows; its frames hold every row, and the
Report publishes them so a reader can look the rest up.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from workbench.reports.paginator import LABEL, TABLE_HEAD, TABLE_ROW, Paginator, Panel
from workbench.reports.report import Report, Section, Summary
from workbench.utils.metrics_utils import performance_breakout
from workbench.utils.archive import Archive, ArchiveRead
from workbench.utils.pandas_utils import DataQuality, KeyedComparison
from workbench.utils.plots import data as data_charts
from workbench.utils.plots import performance as charts
from workbench.utils.plots import report as pages

# Rows of a detail frame shown on a page; the rest are in the published frame
EXAMPLE_ROWS = 8

# Columns whose outliers get a box plot on the data quality page
OUTLIER_BOXES = 3

# Beside the box plots: where they start, the pitch between them, and the most findings listed
BOXES_X, BOX_PITCH, FINDINGS_BESIDE_BOXES = 0.58, 0.115, 12

# Rows shown behind each data quality warning
EXAMPLE_ROWS_PER_WARNING = 10

# Columns shown beside the key in a one-sided example table
EXTRA_COLUMNS = 3

SETS_DIFFER = "several rows share a key, and the two sides do not hold the same values"

# How a repeated key was judged, by `compare_keyed`'s `repeats`
REPEATS = {
    "exact": "agree when both sides hold the same rows: the same count with the same values",
    "values": "agree when every value is on both sides, however many rows carry it",
}

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

    about = [about] if about else []

    def build(report: Report):
        flow = Paginator(report, f"Model performance: {target}", f"{int(overall['support']):,} rows scored.")
        flow.notes(about)
        stats = [("R2", _num(overall["r2"])), ("RMSE", _num(overall["rmse"])), ("MAE", _num(overall["mae"]))]
        stats += [("Spearman", _num(overall["spearmanr"])), ("Rows", f"{int(overall['support']):,}")]
        if log:
            stats += [("GMFE", _num(overall["gmfe"]))]
            stats += [(f"Within {fold}", f"{overall[f'within_{fold}']:.0%}") for fold in FOLDS]
        flow.tiles(stats)
        flow.para(LOG_METRIC_NOTES if log else METRIC_NOTES, size=8.6)

        scatter_title = f"{overall['within_2x']:.0%} of predictions within 2x" if log else f"R2 {overall['r2']:.2f}"
        groups = df[by] if by else None
        median = detail[error].median()
        flow.charts(
            0.33,
            Panel(
                0.10,
                0.37,
                lambda ax: charts.pred_vs_actual(
                    df[target], df[prediction], groups=groups, log=log, unit=unit, title=scatter_title, ax=ax
                ),
            ),
            Panel(
                0.58,
                0.35,
                lambda ax: charts.residuals(
                    df[target], df[prediction], log=log, unit=unit, title=f"Median error {median:+.3g}", ax=ax
                ),
            ),
            above=0.07,
            below=0.06,
        )

        flow.heading("Where the model misses")
        if by:
            metric = "gmfe" if log else "mae"
            within = [f"within_{fold}" for fold in FOLDS]
            columns = ["group", "support", "r2", "rmse"] + (["gmfe"] + within if log else ["mae"])
            shown = breakout.assign(**{c: breakout[c].map("{:.0%}".format) for c in within}) if log else breakout
            flow.label(f"By {by}")
            flow.table(columns, _rows(shown, columns, limit=len(shown)), hl_row=0)
            scores = breakout.iloc[1:].set_index("group")[metric]
            worst = scores.idxmax() if len(scores) else ""
            bars_title = f"{worst} has the highest {metric.upper()}"
            flow.charts(
                0.026 * len(scores) + 0.02,
                Panel(
                    0.24,
                    0.50,
                    lambda ax: charts.breakout_bars(scores, overall[metric], metric.upper(), bars_title, ax=ax),
                ),
                below=0.06,
            )
            flow.where(breakout_frame, len(breakout))
        flow.label("Largest errors")
        columns = list(detail.columns)
        flow.table(columns, _rows(detail, columns, limit=10), frame=rows_frame, count=len(detail))
        return flow.figs

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
    return Section(pages=[build], frames=frames, summary=card, about=about)


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

    has_examples = bool(warnings) and not rows.empty

    def build(report: Report):
        flow = Paginator(report, title, verdict)
        flow.notes(about)
        if findings.empty:
            return flow.figs
        boxed = list(result.boxes["column"].head(OUTLIER_BOXES))
        shown = findings.drop(columns="count").fillna("")
        if not boxed:  # the findings take the page's width, with their detail
            shown = shown.assign(detail=shown["detail"].map(lambda d: _clip(d, 50)))
            columns = ["check", "column", "severity", "value", "detail"]
            flow.label("Findings, worst first")
            flow.table(columns, _rows(shown, columns, limit=30, width=60), frame=findings_frame, count=len(findings))
        else:  # the findings on the left, a box per column with outliers on the right
            columns = ["check", "column", "severity", "value"]
            listed = _rows(shown, columns, limit=FINDINGS_BESIDE_BOXES, width=24)
            box_height = 0.2

            def beside(fig, y):
                pages.sec(fig, y, "Findings, worst first")
                below = pages.table(fig, y - LABEL, columns, listed, x1=BOXES_X - 0.04)
                for place, column in enumerate(boxed):
                    ax = fig.add_axes([BOXES_X + 0.03 + BOX_PITCH * place, y - 0.02 - box_height, 0.07, box_height])
                    data_charts.outlier_box(result, column, column, ax=ax)
                return min(below, y - 0.02 - box_height - 0.06)

            table_height = LABEL + TABLE_HEAD + len(listed) * TABLE_ROW
            flow.block(max(table_height, 0.02 + box_height + 0.06), beside)
            flow.where(findings_frame, len(findings), shown=len(listed))
        if has_examples:
            each = EXAMPLE_ROWS_PER_WARNING
            warned = findings[findings["severity"] == "warn"][["check", "column"]]
            behind = rows.merge(warned, on=["check", "column"])
            picks = behind.groupby(["check", "column"], sort=False, dropna=False).head(each)
            columns = ["check", "column"] + key_cols + ["value"]
            flow.heading(f"Rows to look at: up to {each} behind each warning")
            # Say rows were held back only when a warning has more than are shown
            held_back = len(picks) < len(behind)
            cells = _rows(picks.fillna(""), columns, limit=len(picks), width=90)
            flow.table(columns, cells, frame=rows_frame, count=len(rows), sample=held_back)
        return flow.figs

    headline = verdict
    if len(per_column):
        headline += f" {per_column.idxmax()} has the most flagged rows ({int(per_column.max()):,})."
    card = Summary(
        title,
        headline,
        [("Warnings", warnings), ("Minor", minor), ("Cells flagged", f"{flagged:,}")],
        (lambda ax: data_charts.quality_list(result, most=4, ax=ax)) if len(per_column) else None,
        hero=(str(warnings), "warning" if warnings == 1 else "warnings"),
        needs_action=bool(warnings),
    )
    frames = {findings_frame: findings, rows_frame: rows}
    return Section(pages=[build], frames=frames, summary=card, about=about)


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

    has_examples = bool(open_items) or any(not frame.empty for *_, frame in aside)

    def build(report: Report):
        flow = Paginator(report, title, verdict)
        flow.notes([left, right])
        stats = [("Agree", counts["agree"]), ("Mismatched", counts["mismatched"])]
        stats += [(f"{left} only", counts["left_only"]), (f"{right} only", counts["right_only"])]
        stats += [("Repeated keys differ", counts["repeated_differ"])]
        flow.tiles([(k, f"{v:,}") for k, v in stats])
        flow.table(["how rows were compared", ""], method, aligns=["left", "left"])
        if by is None:
            overlap = Panel(
                pages.L,
                pages.R - pages.L,
                lambda ax: data_charts.overlap_venn(result, left, right, locations=locations, ax=ax),
            )
            flow.charts(0.2, overlap, above=0.0, below=0.025)
        else:  # the breakdown of one-sided rows on the left, the overlap on the right
            breakdown = Panel(
                pages.L + 0.07,
                0.36,
                lambda ax: data_charts.one_sided_bars(
                    result, by, left, right, title=f"Rows on one side only, by {by}", ax=ax
                ),
                height=0.145,
                lift=0.015,
            )
            overlap = Panel(
                0.54,
                pages.R - 0.54,
                lambda ax: data_charts.overlap_venn(result, left, right, locations=locations, location_chars=36, ax=ax),
            )
            flow.charts(0.2, breakdown, overlap, above=0.0, below=0.025)

        fine = [["agree", f"{counts['agree']:,} pairs", "the two sides hold the same values"]]
        for _, source, other, frame in aside:
            if not frame.empty:
                meaning = f"{source} rows with a {' or '.join(result.scope)} {other} never holds"
                fine.append([f"not expected in {other}", f"{len(frame):,} rows", meaning])
        buckets = {"aligns": ["left", "right", "left"]}
        flow.label("Fine")
        flow.table(["bucket", "size", "meaning"], fine, **buckets)
        flow.gap(0.004)
        flow.label("Needs a look")
        needing = [[label, shown or f"{len(frame):,} rows", meaning] for _, label, frame, shown, meaning in look]
        flow.table(["bucket", "size", "meaning"], needing, **buckets)
        if not has_examples:
            return flow.figs

        flow.heading("Rows to look at")
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
            flow.label(f"{label}: {meaning}", short=label)
            cells = _rows(frame, columns, limit=max_rows, width=22)
            flow.table(columns, cells, frame=f"{name}_{suffix}", count=len(frame))
        for suffix, source, other, frame in aside:
            if frame.empty:
                continue
            reasons = frame.groupby("scope_reason").size().sort_values(ascending=False, kind="stable")
            cells = [[reason, f"{count:,}"] for reason, count in reasons.items()]
            label = f"Not expected in {other}"
            flow.label(f"{label}: {source} rows it holds no counterpart for", pages.MUTED, short=label)
            flow.table(["reason", "rows"], cells, frame=f"{name}_{suffix}", count=len(frame), sample=False)
        return flow.figs

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
    return Section(pages=[build], frames=frames, summary=card, about=[left, right])


def archive(
    result: ArchiveRead,
    source: Archive,
    title: Optional[str] = None,
    name: str = "archive",
    card_size: str = "half_height",
) -> Section:
    """How an Archive was read: where it is, how many of its files gave rows, and which could not be read.

    Args:
        result (ArchiveRead): Output of `Archive.read`.
        source (Archive): The Archive that was read.
        title (Optional[str]): Page headline. Defaults to "How the <name> was read".
        name (str): Prefix for this section's frames. Defaults to "archive".
        card_size (str): The size of its card on the summary page: "half_height" (the default),
            "half_width" or "full".

    Returns:
        Section: The page, and the frame `<name>_unreadable` (every file the reader raised on).
    """
    title = title or f"How the {source.name} was read"
    unreadable, unreadable_frame = result.unreadable, f"{name}_unreadable"
    read = result.files - len(unreadable)
    verdict = f"{read:,} of {result.files:,} files read, giving {len(result.rows):,} rows."
    if len(unreadable):
        verdict += f" {len(unreadable):,} could not be read."

    def build(report: Report):
        flow = Paginator(report, title, verdict)
        flow.notes([source.name])
        stats = [("Files", f"{result.files:,}"), ("Read", f"{read:,}"), ("Unreadable", f"{len(unreadable):,}")]
        flow.tiles(stats + [("Rows", f"{len(result.rows):,}")])
        where = [[f"{source.name} is at", source.location], ["files", source.pattern]]
        flow.table(["how it was read", ""], where, aligns=["left", "left"])
        if len(unreadable):
            flow.label("Files that could not be read")
            cells = _rows(unreadable, ["source_file", "reason"], limit=EXAMPLE_ROWS_PER_WARNING, width=70)
            flow.table(["file", "reason"], cells, frame=unreadable_frame, count=len(unreadable))
        return flow.figs

    card = Summary(
        title,
        verdict,
        [("Files", f"{result.files:,}"), ("Unreadable", f"{len(unreadable):,}"), ("Rows", f"{len(result.rows):,}")],
        hero=(f"{read:,}", f"of {result.files:,} files read"),
        needs_action=bool(len(unreadable)),
        size=card_size,
    )
    return Section(pages=[build], frames={unreadable_frame: unreadable}, summary=card, about=[source.name])
