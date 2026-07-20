"""Helpers for champion/challenger contest reports (Reports() /contests subtree)."""

CONTEST_PREFIX = "/contests/"


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
