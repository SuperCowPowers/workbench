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

# A contest is "contested" when the best real challenger is better than the champion, or at
# most this many percent worse, on the primary metric (percent of the champion's value, since
# Δ is an absolute difference). Catches both a close race and a challenger that beats the
# champion but wasn't promoted (a blocked or broken promotion pipeline).
CONTESTED_PCT = -1.0

# The arbiter promotes by freezing a copy of a challenger, so the champion's own twin sits in
# the roster at Δ≈0 and would make every promoted contest contested forever. Challengers this
# close to the champion are that copy (float noise, not a real difference) and are skipped.
TWIN_EPS = 1e-6


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
            columns [model, role, framework, endpoint, created, <metrics interleaved with Δ
            vs champion>, inference_run, timestamp, contested]. Champion Δ columns are 0
            (delta vs itself); framework is the model's framework, with multi-task models
            (list target) reported as "multi-task"; created is the model's creation time;
            contested is the contest-level flag (see CONTESTED_PCT), repeated on every row.
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

    # Only the models that made the report: challengers without metrics were dropped above,
    # and framework/created both cost metadata reads we'd otherwise throw away.
    in_report = set(report["model"])
    models = {m.name: m for m in [champion, *challengers] if m.name in in_report}
    report.insert(2, "framework", report["model"].map({name: _framework(m) for name, m in models.items()}))
    report.insert(3, "endpoint", endpoint_name)
    created = report["model"].map({name: m.created() for name, m in models.items()})
    report.insert(4, "created", pd.to_datetime(created, utc=True))
    delta_cols = [col for col in report.columns if col.startswith("Δ")]
    report.loc[report["role"] == "champion", delta_cols] = 0.0
    report["inference_run"] = inference_run
    report["timestamp"] = pd.Timestamp.now(tz="UTC")
    report["contested"] = _contested(champ_row, chall_rows)
    return report.reset_index(drop=True)


def _contested(champ_row: pd.DataFrame, chall_rows: pd.DataFrame) -> bool:
    """Is the contest contested? True when the best real challenger beats the champion, or is
    at most CONTESTED_PCT percent worse, on the primary metric (rmse for regressors, f1 for
    classifiers). The champion's own frozen copy (Δ within TWIN_EPS) is skipped. Δ is
    absolute, so the percent is taken against the champion's value."""
    if champ_row.empty or chall_rows.empty:
        return False
    if "rmse" in champ_row.columns:
        primary = "rmse"
    elif "f1" in champ_row.columns:
        primary = "f1"
    else:
        return False
    if f"Δ{primary}" not in chall_rows.columns:
        return False
    champ_value = champ_row.iloc[0][primary]
    if pd.isna(champ_value) or champ_value == 0:
        return False

    # Challengers are ranked best-first, so the first non-twin is the best real challenger
    deltas = chall_rows[f"Δ{primary}"]
    real = deltas[deltas.notna() & (deltas.abs() > TWIN_EPS)]
    if real.empty:
        return False
    return bool(real.iloc[0] / abs(champ_value) * 100 >= CONTESTED_PCT)


def _framework(model) -> str:
    """The model's framework for report rows. A list target means multi-task (checked first,
    so a multi-task model reports as such even when it also carries descriptors). A chemprop
    model with more than the SMILES column means hybrid: a graph model fed extra descriptors."""
    try:
        if isinstance(model.target(), list):
            return "multi-task"
        framework = model.model_framework.value
        if framework == "chemprop" and len(model.features() or []) > 1:
            return "hybrid"
        return framework
    except Exception as e:
        log.warning(f"Could not determine framework for {model.name}: {e}")
        return "unknown"


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
