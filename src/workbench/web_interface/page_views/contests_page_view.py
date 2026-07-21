"""ContestsPageView pulls the published contest reports from Reports()"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.api import Reports
from workbench.cached.cached_meta import CachedMeta
from workbench.utils.pipeline_utils import endpoint_group_paths
from workbench.utils.contest_utils import RECENT_CHANGE_HOURS


class ContestsPageView(PageView):
    def __init__(self):
        """ContestsPageView pulls the champion/challenger contest reports (/contests/*)"""
        # Call SuperClass Initialization
        super().__init__()

        # Reports (S3-backed) for the contest data; CachedMeta only for the pipeline
        # hierarchy (group paths). No Model/Endpoint constructions.
        self.reports = Reports()
        self.meta = CachedMeta()

        # One entry per contest: {"group": [path], "recent_change": bool, "rows": [row dicts]}
        # (champion row first, challengers ranked)
        self.contest_data = []

    def refresh(self):
        """Refresh the contest reports from the /contests subtree (parallel S3 gets)"""
        locations = sorted(loc for loc in self.reports.list() if loc.startswith("/contests/"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            dfs = pool.map(self.reports.get, locations)

        groups = endpoint_group_paths(self.meta.pipelines())
        now = pd.Timestamp.now(tz="UTC")
        contests = []
        for df in dfs:
            if df is None or df.empty:
                continue
            df = df.copy()
            df["timestamp"] = df["timestamp"].apply(lambda t: t.isoformat())
            endpoint = df["endpoint"].iloc[0]

            # Champion created (promoted) inside the window -> the card's "recent change" badge
            champion = df[df["role"] == "champion"]
            recent_change = not champion.empty and (now - champion["created"].iloc[0]) < timedelta(
                hours=RECENT_CHANGE_HOURS
            )
            df["created"] = df["created"].dt.strftime("%Y-%m-%d")
            contests.append(
                {
                    "group": groups.get(endpoint, []),
                    "recent_change": bool(recent_change),
                    "rows": df.where(df.notna(), None).to_dict("records"),
                }
            )
        self.contest_data = contests

    def contests(self) -> list:
        """Get the published contest reports

        Returns:
            list: One entry per contest: {"group": [pipeline group path], "recent_change":
                bool (champion promoted within RECENT_CHANGE_HOURS), "rows": [row dicts]}
        """
        return self.contest_data


if __name__ == "__main__":
    # Exercising the ContestsPageView
    from pprint import pprint

    contests_view = ContestsPageView()
    contests_view.refresh()
    pprint(contests_view.contests())
