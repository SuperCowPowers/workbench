"""Model Comparison: Compare inference metrics between two models (e.g. champion vs challenger)"""

from __future__ import annotations

import logging
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from workbench.api import Model

# Set up the log
log = logging.getLogger("workbench")

# Metrics where a lower value is better; everything else (r2, spearmanr, precision,
# recall, f1, roc_auc, ...) is higher-is-better. Support falls through to the plain
# b - a difference, which is what we want for a count.
LOWER_IS_BETTER = {"rmse", "mae", "medae"}


def model_comparison(model_a: Model, model_b: Model, inference_run: str = "default") -> Optional[pd.DataFrame]:
    """Compare the inference metrics of two models.

    For classifiers the comparison uses the 'all' summary row (the same row the
    promotion arbiter ranks on); regressors use their single metrics row.

    Args:
        model_a (Model): The reference model (e.g. the champion)
        model_b (Model): The model compared against it (e.g. a challenger)
        inference_run (str, optional): The inference run to compare. Defaults to "default".

    Returns:
        pd.DataFrame: Three rows indexed [model_a.name, model_b.name, "delta"], one column
            per metric. The delta row is the metrics-aware improvement of model_b over
            model_a: positive always means model_b is better (see LOWER_IS_BETTER).
            None if either model has no metrics for the inference run.
    """
    rows = []
    for model in (model_a, model_b):
        df = model.get_inference_metrics(inference_run)
        if df is None or df.empty:
            log.warning(f"No inference metrics for {model.name} run '{inference_run}'")
            return None
        rows.append(_metrics_row(df, model.name))
    row_a, row_b = rows

    # Compare on the shared metric columns (a regressor/classifier mismatch shares none)
    shared = [col for col in row_a.index if col in row_b.index]
    if not shared:
        log.warning(f"{model_a.name} and {model_b.name} have no metrics in common")
        return None
    row_a, row_b = row_a[shared], row_b[shared]

    delta = pd.Series(
        {col: (row_a[col] - row_b[col]) if col in LOWER_IS_BETTER else (row_b[col] - row_a[col]) for col in shared},
        name="delta",
    )
    return pd.DataFrame([row_a, row_b, delta])


def prediction_comparison(model_a: Model, model_b: Model, inference_run: str = "default") -> Optional[pd.DataFrame]:
    """Concatenated inference predictions for two models, labeled by a 'model' column.

    One row per (model, compound) prediction, with the original prediction columns
    plus 'model' to distinguish the two point sets (e.g. for overlay plotting).

    Args:
        model_a (Model): The reference model (e.g. the champion)
        model_b (Model): The model compared against it (e.g. a challenger)
        inference_run (str, optional): The inference run to compare. Defaults to "default".

    Returns:
        pd.DataFrame: Both models' predictions stacked, with a 'model' column.
            None if either model has no predictions for the inference run.
    """
    dfs = []
    for model in (model_a, model_b):
        df = model.get_inference_predictions(inference_run)
        if df is None or df.empty:
            log.warning(f"No inference predictions for {model.name} run '{inference_run}'")
            return None
        df = df.copy()
        df["model"] = model.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def rank_models(models: list, inference_run: str = "default") -> pd.DataFrame:
    """Rank models by their primary metric: rmse (low to high) for regressors,
    'all' row f1 (high to low) for classifiers.

    Args:
        models (list[Model]): The models to rank
        inference_run (str, optional): The inference run to rank on. Defaults to "default".

    Returns:
        pd.DataFrame: One metrics row per model (indexed by model name), best first.
            Models without metrics for the run are logged and skipped.
    """
    rows = []
    for model in models:
        df = model.get_inference_metrics(inference_run)
        if df is None or df.empty:
            log.warning(f"No inference metrics for {model.name} run '{inference_run}', skipping")
            continue
        rows.append(_metrics_row(df, model.name))
    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    if "rmse" in table.columns:
        return table.sort_values("rmse")
    if "f1" in table.columns:
        return table.sort_values("f1", ascending=False)
    return table


def contest_ranking(champion: Model, challengers: list, inference_run: str = "default") -> Optional[pd.DataFrame]:
    """Rank the challengers of a contest, with metrics-aware deltas against the champion.

    Args:
        champion (Model): The champion model (the delta reference)
        challengers (list[Model]): The challenger models to rank
        inference_run (str, optional): The inference run to compare. Defaults to "default".

    Returns:
        pd.DataFrame: rank_models() of the challengers with a Δ column after each metric
            (positive = challenger better than champion, see LOWER_IS_BETTER; support has
            no Δ). If the champion has no metrics for the inference run, the ranking is
            returned without Δ columns.
    """
    ranked = rank_models(challengers, inference_run)

    champ_df = champion.get_inference_metrics(inference_run)
    if champ_df is None or champ_df.empty:
        log.warning(f"No inference metrics for champion {champion.name} run '{inference_run}': no Δ columns")
        return ranked
    champ_row = _metrics_row(champ_df, champion.name)
    ordered = []
    for col in ranked.columns:
        ordered.append(col)
        if col in champ_row.index and col != "support":
            delta = (champ_row[col] - ranked[col]) if col in LOWER_IS_BETTER else (ranked[col] - champ_row[col])
            ranked[f"Δ{col}"] = delta
            ordered.append(f"Δ{col}")
    return ranked[ordered]


def contest_report(
    champion: Model, challengers: list, endpoint_name: str, inference_run: str = "full_cross_fold"
) -> Optional[pd.DataFrame]:
    """The publishable contest report: champion + ranked challengers in one table.

    Args:
        champion (Model): The model currently serving the contested endpoint
        challengers (list[Model]): The challenger models
        endpoint_name (str): The contested endpoint (recorded in the report)
        inference_run (str, optional): The inference run to compare. Defaults to "full_cross_fold".

    Returns:
        pd.DataFrame: One row per model (champion first, then challengers best-first) with
            columns [model, role, endpoint, <metrics interleaved with Δ vs champion>,
            inference_run, timestamp]. Champion Δ columns are 0 (delta vs itself).
            Models without metrics are skipped; None if no model has metrics.
    """
    champ_row = rank_models([champion], inference_run)
    chall_rows = contest_ranking(champion, challengers, inference_run)
    if champ_row.empty and chall_rows.empty:
        log.warning(f"No metrics for any model in the '{endpoint_name}' contest: no report")
        return None

    # Champion first, challengers best-first, columns in the interleaved metric/Δ order
    cols = list(chall_rows.columns) if not chall_rows.empty else list(champ_row.columns)
    report = pd.concat([champ_row, chall_rows])[cols]
    report.insert(0, "model", report.index)
    report.insert(1, "role", ["champion"] * len(champ_row) + ["challenger"] * len(chall_rows))
    report.insert(2, "endpoint", endpoint_name)
    delta_cols = [col for col in report.columns if col.startswith("Δ")]
    report.loc[report["role"] == "champion", delta_cols] = 0.0
    report["inference_run"] = inference_run
    report["timestamp"] = pd.Timestamp.now(tz="UTC")
    return report.reset_index(drop=True)


def contest_summaries() -> pd.DataFrame:
    """One row per published contest report (the card-grid view of /contests/*).

    Returns:
        pd.DataFrame: Columns [endpoint, champion, challengers, top_challenger,
            primary_metric, top_delta, contested, inference_run, timestamp], newest
            first. 'contested' means the top challenger beats the champion on the
            primary metric (rmse for regressors, f1 for classifiers). Empty if no
            contest reports are published.
    """
    from concurrent.futures import ThreadPoolExecutor
    from workbench.api import Reports

    reports = Reports()
    locations = [loc for loc in reports.list() if loc.startswith("/contests/")]
    if not locations:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=8) as pool:
        contests = pool.map(reports.get, locations)

    rows = []
    for df in contests:
        if df is None or df.empty:
            continue
        champions = df[df["role"] == "champion"]
        challengers = df[df["role"] == "challenger"]
        primary = "rmse" if "rmse" in df.columns else "f1"
        top = challengers.iloc[0] if not challengers.empty else None
        top_delta = top[f"Δ{primary}"] if top is not None and f"Δ{primary}" in df.columns else None
        rows.append(
            {
                "endpoint": df["endpoint"].iloc[0],
                "champion": champions["model"].iloc[0] if not champions.empty else None,
                "challengers": len(challengers),
                "top_challenger": top["model"] if top is not None else None,
                "primary_metric": primary,
                "top_delta": top_delta,
                "contested": bool(top_delta > 0) if top_delta is not None else None,
                "inference_run": df["inference_run"].iloc[0],
                "timestamp": df["timestamp"].iloc[0],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False, ignore_index=True)


def _metrics_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    """The single metrics row to compare on: the 'all' summary row for classifiers
    (per-class label column dropped), the first row for regressors."""
    all_row = df[df.eq("all").any(axis=1)]
    if not all_row.empty:
        df = all_row
    row = df.iloc[0].apply(pd.to_numeric, errors="coerce").dropna()
    row.name = model_name
    return row


if __name__ == "__main__":
    # Exercise the model comparison utility (champion vs challenger)
    from workbench.cached.cached_model import CachedModel

    print("*** Regression: aqsol-regression (champion) vs aqsol-regression-1 ***")
    comparison = model_comparison(CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "full_cross_fold")
    print(comparison)

    print("\n*** Classification: aqsol-class (champion) vs aqsol-class-1 ***")
    comparison = model_comparison(CachedModel("aqsol-class"), CachedModel("aqsol-class-1"), "full_cross_fold")
    print(comparison)

    print("\n*** Missing inference run returns None ***")
    comparison = model_comparison(CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "no-such-run")
    print(comparison)

    print("\n*** Contest ranking: aqsol-regression challengers ***")
    champion = CachedModel("aqsol-regression")
    challengers = [CachedModel("aqsol-regression-1"), CachedModel("aqsol-regression-2")]
    ranking = contest_ranking(champion, challengers, "full_cross_fold")
    print(ranking)

    print("\n*** Contest ranking: aqsol-class challengers ***")
    champion = CachedModel("aqsol-class")
    challengers = [CachedModel("aqsol-class-1"), CachedModel("aqsol-class-2")]
    ranking = contest_ranking(champion, challengers, "full_cross_fold")
    print(ranking)

    print("\n*** Contest report: aqsol-regression ***")
    champion = CachedModel("aqsol-regression")
    challengers = [CachedModel("aqsol-regression-1"), CachedModel("aqsol-regression-2")]
    report = contest_report(champion, challengers, "aqsol-regression")
    print(report)

    print("\n*** Contest summaries (published /contests/* reports) ***")
    summaries = contest_summaries()
    print(summaries)
