"""ModelComparisonView: the full published contest reports.

Contest reports are published to the Reports store (/contests/*) by the promotion
arbiter. This view is a thin parallel read of that subtree -- no batch computation,
no model construction. The page ships the raw rows to the browser and renders the
rail + comparison table clientside (assets/mc/render.js).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.api import Reports
from workbench.cached.cached_meta import CachedMeta
from workbench.utils.pipeline_utils import endpoint_group_paths

# A contest is flagged "recent change" when its champion was promoted (created) within this
# window -- mirrors the Contests page. Relative to now, so it's computed per refresh.
RECENT_CHANGE_HOURS = 72


class ModelComparisonView(PageView):
    """Every published model contest, as raw report rows (champion + challengers)."""

    def __init__(self):
        """Initialize the ModelComparisonView"""
        super().__init__()
        self.reports = Reports()
        self.meta = CachedMeta()  # for the pipeline-hierarchy grouping (the rail's sections)

        # One entry per contest: {"endpoint": str, "group": [path], "rows": [row dicts]},
        # champion row first. Loaded lazily -- the page's data callback calls refresh().
        self.contest_data = []

    def refresh(self):
        """Re-read every /contests/* report (parallel S3 gets)"""
        locations = sorted(loc for loc in self.reports.list() if loc.startswith("/contests/"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            dfs = pool.map(self.reports.get, locations)

        # Endpoint -> pipeline group path (the same grouping the ML Pipelines / Contests
        # pages use); the rail groups contests by the top level of this path.
        groups = endpoint_group_paths(self.meta.pipelines())
        now = pd.Timestamp.now(tz="UTC")

        contests = []
        for df in dfs:
            if df is None or df.empty:
                continue
            df = df.copy()

            # Champion promoted inside the window -> "recent change" flag (computed off the
            # raw Timestamp, before the datetime columns are stringified below).
            champion = df[df["role"] == "champion"]
            recent_change = not champion.empty and (now - champion["created"].iloc[0]) < timedelta(
                hours=RECENT_CHANGE_HOURS
            )

            # dcc.Store data is JSON: stringify datetime columns, then NaN -> None.
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].apply(lambda t: t.isoformat() if pd.notna(t) else None)
            endpoint = df["endpoint"].iloc[0]
            contests.append(
                {
                    "endpoint": endpoint,
                    "group": groups.get(endpoint, []),
                    "recent_change": bool(recent_change),
                    "rows": df.where(df.notna(), None).to_dict("records"),
                }
            )
        self.contest_data = contests

    def view_data(self) -> list:
        """Get the full contest reports.

        Returns:
            list: One entry per contest: {"endpoint": str, "group": [pipeline group path],
                "recent_change": bool (champion promoted within RECENT_CHANGE_HOURS),
                "rows": [row dicts]} with the champion row first, then challengers ranked
                best-first. Empty if none are published.
        """
        return self.contest_data


if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()
    view = ModelComparisonView()
    view.refresh()
    data = view.view_data()
    print(f"Refresh time: {(datetime.now() - start).total_seconds():.2f}s, contests: {len(data)}")
    for c in data[:3]:
        print(f"  {c['endpoint']}: {len(c['rows'])} rows")
