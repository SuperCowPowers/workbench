"""ContestsPageView pulls the published contest reports from Reports()"""

from concurrent.futures import ThreadPoolExecutor

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.api import Reports
from workbench.cached.cached_meta import CachedMeta


class ContestsPageView(PageView):
    def __init__(self):
        """ContestsPageView pulls the champion/challenger contest reports (/contests/*)"""
        # Call SuperClass Initialization
        super().__init__()

        # Reports (S3-backed) for the contest data; CachedMeta only for the pipeline
        # hierarchy (group paths). No Model/Endpoint constructions.
        self.reports = Reports()
        self.meta = CachedMeta()

        # One entry per contest: {"group": [path], "rows": [row dicts]}
        # (champion row first, challengers ranked)
        self.contest_data = []

    def refresh(self):
        """Refresh the contest reports from the /contests subtree (parallel S3 gets)"""
        locations = sorted(loc for loc in self.reports.list() if loc.startswith("/contests/"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            dfs = pool.map(self.reports.get, locations)

        groups = self._endpoint_groups()
        contests = []
        for df in dfs:
            if df is None or df.empty:
                continue
            df = df.copy()
            df["timestamp"] = df["timestamp"].apply(lambda t: t.isoformat())
            endpoint = df["endpoint"].iloc[0]
            contests.append(
                {"group": groups.get(endpoint, []), "rows": df.where(df.notna(), None).to_dict("records")}
            )
        self.contest_data = contests

    def contests(self) -> list:
        """Get the published contest reports

        Returns:
            list: One entry per contest: {"group": [pipeline group path], "rows": [row dicts]}
        """
        return self.contest_data

    def _endpoint_groups(self) -> dict:
        """{endpoint_name: [group path]} from the ML pipeline hierarchy (same grouping
        as the ML Pipelines page; endpoints not in any pipeline map to no group)"""
        groups = {}

        def walk(nodes, path):
            for g in nodes:
                p = path + [g["name"]]
                for graph in (g.get("pipelines") or {}).values():
                    for node in graph.get("nodes", []):
                        if node.get("type") == "endpoint":
                            groups.setdefault(node["id"].split(":", 1)[-1], p)
                walk(g.get("subgroups") or [], p)

        walk(self.meta.pipelines(), [])
        return groups


if __name__ == "__main__":
    # Exercising the ContestsPageView
    from pprint import pprint

    contests_view = ContestsPageView()
    contests_view.refresh()
    pprint(contests_view.contests())
