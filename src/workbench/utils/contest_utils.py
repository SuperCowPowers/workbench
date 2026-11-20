"""Helpers for champion/challenger contest reports (DFStore /reports/contests subtree)."""

from datetime import timedelta

import pandas as pd

CONTEST_PREFIX = "/contests/"

# A contest gets the "recent change" flag when its champion was created (promoted) within this
# window. Relative to now, so it's computed per call rather than stored in the report. Shared by
# the REPL greeting and the dashboard contests page so the two never drift.
RECENT_CHANGE_HOURS = 72


def contest_store():
    """The store contest reports are published to: a DFStore scoped to /reports.

    Returns:
        DFStore: Locations are relative to /reports, so a contest is "/contests/<endpoint>".
    """
    from workbench.api.df_store import DFStore

    return DFStore(path_prefix="/reports")


def find_contests(model_name: str) -> list:
    """Find the contests a model takes part in.

    Contest membership lives in the published report rows, not on the Model
    object, so it has to be looked up rather than inferred.

    Args:
        model_name (str): The model name (use the promoted name; champions are
            usually the dated copy, e.g. "my-model-260715").

    Returns:
        list: One {"contest", "role", "endpoint"} dict per hit, empty if none.
    """
    reports = contest_store()

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


def contest_summary() -> list:
    """One row per contest for the REPL greeting, most recent first.

    Returns:
        list: One dict per contest, sorted newest-first, with keys "contest"
            (name, prefix stripped), "champion", "challengers" (count),
            "endpoint", "contested" (bool), "recent_change" (bool, champion
            promoted within RECENT_CHANGE_HOURS), and "timestamp".
    """
    reports = contest_store()

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
