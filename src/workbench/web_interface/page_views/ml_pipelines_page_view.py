"""MLPipelinesPageView pulls the ML Pipeline hierarchy from CachedMeta"""

# Workbench Imports
from workbench.web_interface.page_views.page_view import PageView
from workbench.cached.cached_meta import CachedMeta


class MLPipelinesPageView(PageView):
    def __init__(self):
        """MLPipelinesPageView pulls the ML Pipeline hierarchy (category/assay/pipeline graphs)"""
        # Call SuperClass Initialization
        super().__init__()

        # CachedMeta gives us the Redis-cached pipeline hierarchy
        self.meta = CachedMeta()

        # Pipeline hierarchy: top-level groups, each {name, subgroups, pipelines}
        self.hierarchy = []

    def refresh(self):
        """Refresh the pipeline hierarchy from CachedMeta (Redis-cached; hits S3 only when stale)"""
        self.hierarchy = self.meta.pipelines()

    def pipelines(self) -> list:
        """Get the full ML Pipeline hierarchy

        Returns:
            list: Top-level groups, each {name, subgroups, pipelines}
        """
        return self.hierarchy


if __name__ == "__main__":
    # Exercising the MLPipelinesPageView
    from pprint import pprint

    pipelines_view = MLPipelinesPageView()
    pipelines_view.refresh()

    def _walk(groups, depth=0):
        for g in groups:
            pipes = list(g["pipelines"].keys())
            print(f"{'  ' * depth}{g['name']}: {pipes}")
            _walk(g["subgroups"], depth + 1)

    _walk(pipelines_view.pipelines())
    pprint(pipelines_view.pipelines())
