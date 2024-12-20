"""CachedPipeline: Caches the method results for Workbench Pipelines"""

# Workbench Imports
from workbench.api.pipeline import Pipeline
from workbench.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedPipeline(CachedArtifactMixin, Pipeline):
    """CachedPipeline: Caches the method results for Workbench Pipelines

    Note: Cached method values may lag underlying Pipeline changes.

    Common Usage:
        ```python
        my_pipeline = CachedPipeline(name)
        my_pipeline.details()
        my_pipeline.health_check()
        ```
    """

    def __init__(self, pipeline_uuid: str):
        """CachedPipeline Initialization"""
        Pipeline.__init__(self, name=pipeline_uuid)

    @CachedArtifactMixin.cache_result
    def summary(self, **kwargs) -> dict:
        """Retrieve the CachedPipeline Details.

        Returns:
            dict: A dictionary of details about the CachedPipeline
        """
        return super().summary(**kwargs)

    @CachedArtifactMixin.cache_result
    def details(self, **kwargs) -> dict:
        """Retrieve the CachedPipeline Details.

        Returns:
            dict: A dictionary of details about the CachedPipeline
        """
        return super().details(**kwargs)

    @CachedArtifactMixin.cache_result
    def health_check(self, **kwargs) -> dict:
        """Retrieve the CachedPipeline Health Check.

        Returns:
            dict: A dictionary of health check details for the CachedPipeline
        """
        return super().health_check(**kwargs)


if __name__ == "__main__":
    """Exercise the CachedPipeline Class"""
    from pprint import pprint

    # Retrieve an existing Pipeline
    my_pipeline = CachedPipeline("abalone_pipeline_v1")
    pprint(my_pipeline.summary())
    pprint(my_pipeline.details())
    pprint(my_pipeline.health_check())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_pipeline._shutdown()
