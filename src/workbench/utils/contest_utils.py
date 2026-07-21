"""Helpers for champion/challenger contest reports (Reports() /contests subtree)."""

from datetime import timedelta

import pandas as pd

CONTEST_PREFIX = "/contests/"

# A contest gets the "recent change" flag when its champion was created (promoted) within this
# window. Relative to now, so it's computed per call rather than stored in the report. Shared by
# the REPL greeting and the dashboard contests page so the two never drift.
RECENT_CHANGE_HOURS = 72


def find_contests(model_name: str, reports=None) -> list:
    """Find the contests a model takes part in.

    Contest membership lives in the published report rows, not on the Model
    object, so it has to be looked up rather than inferred.

    Args:
        model_name (str): The model name (use the promoted name; champions are
            usually the dated copy, e.g. "my-model-260715").
        reports (Reports): Report store to search; defaults to Reports().

    Returns:
        list: One {"contest", "role", "endpoint"} dict per hit, empty if none.
    """
    if reports is None:
        from workbench.api.reports import Reports

        reports = Reports()

    hits = []
    for location in reports.list():
        if not location.startswith(CONTEST_PREFIX):
            continue
        df = reports.get(location)
        if df is None or "model" not in df.columns:
            continue
        rows = df[df["model"] == model_name]
        if not rows.empty:
            row = rows.iloc[0]
            hits.append(
                {
                    "contest": location,
                    "role": row.get("role"),
                    "endpoint": row.get("endpoint"),
                }
            )
    return hits


def contest_summary(reports=None) -> list:
    """One row per contest for the REPL greeting, most recent first.

    Args:
        reports (Reports): Report store to search; defaults to Reports().

    Returns:
        list: One dict per contest, sorted newest-first, with keys "contest"
            (name, prefix stripped), "champion", "challengers" (count),
            "endpoint", "contested" (bool), "recent_change" (bool, champion
            promoted within RECENT_CHANGE_HOURS), and "timestamp".
    """
    if reports is None:
        from workbench.api.reports import Reports

        reports = Reports()

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for location in reports.list():
        if not location.startswith(CONTEST_PREFIX):
            continue
        df = reports.get(location)
        if df is None or "role" not in df.columns:
            continue
        champ = df[df["role"] == "champion"]

        recent_change = False
        if not champ.empty and "created" in df.columns:
            created = champ["created"].iloc[0]
            recent_change = pd.notna(created) and (now - created) < timedelta(hours=RECENT_CHANGE_HOURS)

        rows.append(
            {
                "contest": location[len(CONTEST_PREFIX) :],
                "champion": champ.iloc[0]["model"] if not champ.empty else None,
                "challengers": int((df["role"] == "challenger").sum()),
                "endpoint": df.iloc[0].get("endpoint"),
                "contested": bool(df.iloc[0].get("contested", False)),
                "recent_change": bool(recent_change),
                "timestamp": df["timestamp"].max() if "timestamp" in df.columns else None,
            }
        )
    rows.sort(key=lambda r: (r["timestamp"] is not None, r["timestamp"]), reverse=True)
    return rows
