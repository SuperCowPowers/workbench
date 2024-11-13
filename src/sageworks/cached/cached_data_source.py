"""CachedDataSource: Caches the method results for SageWorks DataSources"""

from typing import Union

# SageWorks Imports
from sageworks.core.artifacts.athena_source import AthenaSource
from sageworks.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedDataSource(CachedArtifactMixin, AthenaSource):
    """CachedDataSource: Caches the method results for SageWorks DataSources

    Note: Cached method values may lag underlying DataSource changes.

    Common Usage:
        ```python
        my_data = CachedDataSource(name)
        my_data.details()
        my_data.health_check()
        my_data.sageworks_meta()
        ```
    """

    def __init__(self, data_uuid: str, database: str = "sageworks"):
        """CachedDataSource Initialization"""
        AthenaSource.__init__(self, data_uuid=data_uuid, database=database, use_cached_meta=True)

    @CachedArtifactMixin.cache_result
    def summary(self, **kwargs) -> dict:
        """Retrieve the DataSource Details.

        Returns:
            dict: A dictionary of details about the DataSource
        """
        return super().summary(**kwargs)

    @CachedArtifactMixin.cache_result
    def details(self, **kwargs) -> dict:
        """Retrieve the DataSource Details.

        Returns:
            dict: A dictionary of details about the DataSource
        """
        return super().details(**kwargs)

    @CachedArtifactMixin.cache_result
    def health_check(self, **kwargs) -> dict:
        """Retrieve the DataSource Health Check.

        Returns:
            dict: A dictionary of health check details for the DataSource
        """
        return super().health_check(**kwargs)

    @CachedArtifactMixin.cache_result
    def sageworks_meta(self) -> Union[dict, None]:
        """Retrieve the SageWorks Metadata for this DataSource.

        Returns:
            Union[dict, None]: Dictionary of SageWorks metadata for this Artifact
        """
        return super().sageworks_meta()


if __name__ == "__main__":
    """Exercise the CachedDataSource Class"""
    from pprint import pprint

    # Retrieve an existing DataSource
    my_data = CachedDataSource("abalone_data")
    pprint(my_data.summary())
    pprint(my_data.details())
    pprint(my_data.health_check())
    pprint(my_data.sageworks_meta())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_data._shutdown()
