"""Report sections: each turns an analysis result into pages, plus the detail frames behind them.

A section's pages show counts and a few example rows; its frames hold every row, and the
Report publishes them so a reader can look the rest up.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from workbench.reports.report import Report, Section
from workbench.utils.metrics_utils import performance_breakout
from workbench.utils.pandas_utils import DataQuality, KeyedComparison
from workbench.utils.plots import performance as charts
from workbench.utils.plots import report as pages

# Rows of a detail frame shown on a page; the rest are in the published frame
EXAMPLE_ROWS = 8

# Page geometry in figure units: the lowest a body may reach above the footer, a table's
# header, and one table row
BOTTOM, TABLE_HEAD, TABLE_ROW = 0.06, 0.03, 0.0172

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
    return f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value)


def _clip(value, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _rows(df: pd.DataFrame, columns: List[str], limit: int = EXAMPLE_ROWS, width: int = 28) -> List[List[str]]:
    return [[_clip(_num(v), width) for v in row] for row in df[columns].head(limit).itertuples(index=False)]


def _rows_that_fit(height: float, most: int) -> int:
    """How many table rows fit in `height` of the page, under a header: at least 1, at most `most`."""
    return int(min(most, max(1, (height - TABLE_HEAD) // TABLE_ROW)))


def _where(report: Report, fig, y: float, frame: str, count: int) -> float:
    """A muted line saying where every row of a frame is; nothing when the report is unpublished."""
    location = report.location(frame)
    if location is None or count == 0:
        return y
    return pages.para(fig, y, f"All {count:,} rows: DFStore {location}", size=8.2, color=pages.MUTED) - 0.012


def performance(
    df: pd.DataFrame,
    target: str,
    prediction: str = "prediction",
    by: Optional[str] = None,
    key: Union[str, List[str], None] = None,
    log: bool = False,
    unit: Optional[str] = None,
    name: str = "performance",
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
        stats = [("R2", _num(overall["r2"])), ("RMSE", _num(overall["rmse"])), ("MAE", _num(overall["mae"]))]
        stats += [("Spearman", _num(overall["spearmanr"])), ("Rows", f"{int(overall['support']):,}")]
        if log:
            stats += [("GMFE", _num(overall["gmfe"]))]
            stats += [(f"Within {fold}", f"{overall[f'within_{fold}']:.0%}") for fold in FOLDS]
        y = pages.stat_row(fig, pages.L, y, stats)
        y = pages.para(fig, y - 0.006, LOG_METRIC_NOTES if log else METRIC_NOTES, size=8.6)

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
        y = pages.table(fig, y, columns, _rows(detail, columns, limit=_rows_that_fit(y - BOTTOM - 0.03, most=10)))
        _where(report, fig, y - 0.008, rows_frame, len(detail))
        return fig

    return Section(pages=[overview, misses], frames={rows_frame: detail, breakout_frame: breakout})


def data_quality(result: DataQuality, title: str = "Data quality", name: str = "quality") -> Section:
    """Data quality: the findings, then example rows for each warning.

    Args:
        result (DataQuality): Output of `pandas_utils.data_quality`.
        title (str): Page headline. Defaults to "Data quality".
        name (str): Prefix for this section's frames. Defaults to "quality".

    Returns:
        Section: One or two pages, and the frames `<name>_findings` and `<name>_rows`
            (every offending cell).
    """
    findings, rows = result.findings, result.rows
    key_cols = [c for c in rows.columns if c not in ("check", "column", "value")]
    warnings = int((findings["severity"] == "warn").sum())
    notes = int((findings["severity"] == "info").sum())
    findings_frame, rows_frame = f"{name}_findings", f"{name}_rows"

    def summary(report: Report):
        sub = f"{warnings} warnings and {notes} notes." if warnings or notes else "No problems found."
        fig, y = report.new_page(title, sub)
        y = pages.stat_row(
            fig, pages.L, y, [("Warnings", warnings), ("Notes", notes), ("Cells listed", f"{len(rows):,}")]
        )
        if findings.empty:
            return fig
        shown = findings.fillna("").assign(detail=findings["detail"].map(lambda d: _clip(d, 50)))
        y = pages.sec(fig, y - 0.01, "Findings, worst first")
        columns = ["check", "column", "severity", "value", "detail"]
        y = pages.table(fig, y, columns, _rows(shown, columns, limit=30, width=60))
        _where(report, fig, y - 0.008, findings_frame, len(findings))
        return fig

    def examples(report: Report):
        fig, y = report.new_page(f"{title}: rows to look at", "The first rows behind each warning.")
        warned = findings[findings["severity"] == "warn"][["check", "column"]]
        picks = (
            rows.merge(warned, on=["check", "column"]).groupby(["check", "column"], sort=False, dropna=False).head(3)
        )
        columns = ["check", "column"] + key_cols + ["value"]
        y = pages.table(fig, y, columns, _rows(picks.fillna(""), columns, limit=30))
        _where(report, fig, y - 0.008, rows_frame, len(rows))
        return fig

    has_examples = warnings and not rows.empty
    return Section(
        pages=[summary] + ([examples] if has_examples else []), frames={findings_frame: findings, rows_frame: rows}
    )


def comparison(
    result: KeyedComparison,
    left: str = "left",
    right: str = "right",
    title: Optional[str] = None,
    name: str = "comparison",
) -> Section:
    """Data comparison: where every row of two frames went, then the rows that need a look.

    Args:
        result (KeyedComparison): Output of `pandas_utils.compare_keyed`.
        left (str): What to call the left frame in the document.
        right (str): What to call the right frame.
        title (Optional[str]): Page headline. Defaults to "<left> against <right>".
        name (str): Prefix for this section's frames. Defaults to "comparison".

    Returns:
        Section: One or two pages, and a frame per bucket: `<name>_mismatched` (one row per
            disagreeing value), `_left_only`, `_right_only`, `_ambiguous`,
            `_left_out_of_scope`, `_right_out_of_scope`.
    """
    title = title or f"{left} against {right}"
    on, pairs, counts = result.on, result.pairs, result.summary()
    values = [c[: -len("_agree")] for c in pairs.columns if c.endswith("_agree")]
    disagreements = []
    for value in values:
        off = pairs[~pairs[f"{value}_agree"]]
        sides = {left: off[f"{value}_left"].astype(str), right: off[f"{value}_right"].astype(str)}
        disagreements.append(off[on].assign(column=value, **sides))
    mismatched = pd.concat(disagreements, ignore_index=True) if disagreements else pd.DataFrame(columns=on)

    buckets = [
        ("mismatched", mismatched, f"{counts['mismatched']:,} pairs", "one row on each side, values disagree"),
        ("left_only", result.left_only, None, f"in {left}, key not in {right}"),
        ("right_only", result.right_only, None, f"in {right}, key not in {left}"),
        ("ambiguous", result.ambiguous, None, "key repeated on a side, so no single pairing to score"),
        ("left_out_of_scope", result.left_out_of_scope, None, f"in {left}, carrying a value {right} never holds"),
        ("right_out_of_scope", result.right_out_of_scope, None, f"in {right}, carrying a value {left} never holds"),
    ]
    frames = {f"{name}_{bucket}": frame for bucket, frame, _, _ in buckets}

    def overview(report: Report):
        fig, y = report.new_page(title, f"{counts['agree']:,} of {counts['pairs']:,} uniquely keyed pairs agree.")
        stats = [("Pairs", counts["pairs"]), ("Agree", counts["agree"]), ("Mismatched", counts["mismatched"])]
        stats += [(f"{left} only", counts["left_only"]), (f"{right} only", counts["right_only"])]
        stats += [("Ambiguous keys", counts["ambiguous_keys"])]
        y = pages.stat_row(fig, pages.L, y, [(k, f"{v:,}") for k, v in stats])
        note = f"Rows are matched on {', '.join(on)}. Only keys held by one row on each side are scored"
        note += f", on {', '.join(values)}." if values else "."
        y = pages.para(fig, y - 0.006, note, size=8.6)
        y = pages.sec(fig, y - 0.01, "Where every row went")
        rows = [["agree", f"{counts['agree']:,} pairs", "one row on each side, values agree"]]
        rows += [[bucket, shown or f"{len(frame):,} rows", meaning] for bucket, frame, shown, meaning in buckets]
        pages.table(fig, y, ["bucket", "size", "meaning"], rows, aligns=["left", "right", "left"])
        return fig

    def examples(report: Report):
        fig, y = report.new_page(f"{title}: rows to look at")
        filled = sum(not frame.empty for _, frame, _, _ in buckets)
        limit = _rows_that_fit((y - BOTTOM) / filled - 0.065, most=6)
        for bucket, frame, _, meaning in buckets:
            if frame.empty:
                continue
            columns = list(frame.columns) if bucket in ("mismatched", "ambiguous") else on
            if bucket == "mismatched":
                numeric = frame[[left, right]].apply(pd.to_numeric, errors="coerce")
                frame = frame.assign(
                    **{
                        side: numeric[side].map(_num).where(numeric[side].notna(), frame[side])
                        for side in (left, right)
                    }
                )
            y = pages.sec(fig, y, f"{bucket.replace('_', ' ')}: {meaning}")
            y = pages.table(fig, y, columns, _rows(frame, columns, limit=limit))
            y = _where(report, fig, y - 0.004, f"{name}_{bucket}", len(frame)) - 0.008
        return fig

    has_examples = any(not frame.empty for _, frame, _, _ in buckets)
    return Section(pages=[overview] + ([examples] if has_examples else []), frames=frames)
