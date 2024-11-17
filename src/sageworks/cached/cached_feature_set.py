"""CachedFeatureSet: Caches the method results for SageWorks FeatureSets"""

from typing import Union

# SageWorks Imports
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedFeatureSet(CachedArtifactMixin, FeatureSetCore):
    """CachedFeatureSet: Caches the method results for SageWorks FeatureSets

    Note: Cached method values may lag underlying FeatureSet changes.

    Common Usage:
        ```python
        my_features = CachedFeatureSet(name)
        my_features.details()
        my_features.health_check()
        my_features.sageworks_meta()
        ```
    """

    def __init__(self, feature_set_uuid: str, database: str = "sageworks"):
        """CachedFeatureSet Initialization"""
        FeatureSetCore.__init__(self, feature_set_uuid=feature_set_uuid, use_cached_meta=True)

    @CachedArtifactMixin.cache_result
    def summary(self, **kwargs) -> dict:
        """Retrieve the FeatureSet Details.

        Returns:
            dict: A dictionary of details about the FeatureSet
        """
        return super().summary(**kwargs)

    @CachedArtifactMixin.cache_result
    def details(self, **kwargs) -> dict:
        """Retrieve the FeatureSet Details.

        Returns:
            dict: A dictionary of details about the FeatureSet
        """
        return super().details(**kwargs)

    @CachedArtifactMixin.cache_result
    def health_check(self, **kwargs) -> dict:
        """Retrieve the FeatureSet Health Check.

        Returns:
            dict: A dictionary of health check details for the FeatureSet
        """
        return super().health_check(**kwargs)

    @CachedArtifactMixin.cache_result
    def sageworks_meta(self) -> Union[str, None]:
        """Retrieve the SageWorks Metadata for this DataSource.

        Returns:
            Union[dict, None]: Dictionary of SageWorks metadata for this Artifact
        """
        return super().sageworks_meta()


if __name__ == "__main__":
    """Exercise the CachedFeatureSet Class"""
    from pprint import pprint

    # Retrieve an existing FeatureSet
    my_features = CachedFeatureSet("abalone_features")
    pprint(my_features.summary())
    pprint(my_features.details())
    pprint(my_features.health_check())
    pprint(my_features.sageworks_meta())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_features._shutdown()