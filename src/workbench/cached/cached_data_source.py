"""CachedDataSource: Caches the method results for Workbench DataSources"""

from typing import Union

import pandas as pd

# Workbench Imports
from workbench.core.artifacts.athena_source import AthenaSource
from workbench.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedDataSource(CachedArtifactMixin, AthenaSource):
    """CachedDataSource: Caches the method results for Workbench DataSources

    Note: Cached method values may lag underlying DataSource changes.

    Common Usage:
        ```python
        my_data = CachedDataSource(name)
        my_data.details()
        my_data.health_check()
        my_data.workbench_meta()
        ```
    """

    def __init__(self, data_uuid: str, database: str = "workbench"):
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
    def workbench_meta(self) -> Union[dict, None]:
        """Retrieve the Workbench Metadata for this DataSource.

        Returns:
            Union[dict, None]: Dictionary of Workbench metadata for this Artifact
        """
        return super().workbench_meta()

    def smart_sample(self) -> pd.DataFrame:
        """Retrieve the Smart Sample for this DataSource.

        Returns:
            pd.DataFrame: The Smart Sample DataFrame
        """
        return super().smart_sample()


if __name__ == "__main__":
    """Exercise the CachedDataSource Class"""
    from pprint import pprint

    # Retrieve an existing DataSource
    my_data = CachedDataSource("abalone_data")
    pprint(my_data.summary())
    pprint(my_data.details())
    pprint(my_data.health_check())
    pprint(my_data.workbench_meta())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_data._shutdown()
