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
            no Δ). None if the champion has no metrics for the inference run.
    """
    champ_df = champion.get_inference_metrics(inference_run)
    if champ_df is None or champ_df.empty:
        log.warning(f"No inference metrics for champion {champion.name} run '{inference_run}'")
        return None
    champ_row = _metrics_row(champ_df, champion.name)

    ranked = rank_models(challengers, inference_run)
    ordered = []
    for col in ranked.columns:
        ordered.append(col)
        if col in champ_row.index and col != "support":
            delta = (champ_row[col] - ranked[col]) if col in LOWER_IS_BETTER else (ranked[col] - champ_row[col])
            ranked[f"Δ{col}"] = delta
            ordered.append(f"Δ{col}")
    return ranked[ordered]


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
    comparison = model_comparison(
        CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "full_cross_fold"
    )
    print(comparison)

    print("\n*** Classification: aqsol-class (champion) vs aqsol-class-1 ***")
    comparison = model_comparison(CachedModel("aqsol-class"), CachedModel("aqsol-class-1"), "full_cross_fold")
    print(comparison)

    print("\n*** Missing inference run returns None ***")
    comparison = model_comparison(
        CachedModel("aqsol-regression"), CachedModel("aqsol-regression-1"), "no-such-run"
    )
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
