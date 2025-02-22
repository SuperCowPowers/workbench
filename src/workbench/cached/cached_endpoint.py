"""CachedEndpoint: Caches the method results for Workbench Endpoints"""

from typing import Union

# Workbench Imports
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedEndpoint(CachedArtifactMixin, EndpointCore):
    """CachedEndpoint: Caches the method results for Workbench Endpoints

    Note: Cached method values may lag underlying Endpoint changes.

    Common Usage:
        ```python
        my_endpoint = CachedEndpoint(name)
        my_endpoint.details()
        my_endpoint.health_check()
        my_endpoint.workbench_meta()
        ```
    """

    def __init__(self, endpoint_uuid: str):
        """CachedEndpoint Initialization"""
        EndpointCore.__init__(self, endpoint_uuid=endpoint_uuid, use_cached_meta=True)

    @CachedArtifactMixin.cache_result
    def summary(self, **kwargs) -> dict:
        """Retrieve the CachedEndpoint Details.

        Returns:
            dict: A dictionary of details about the CachedEndpoint
        """
        return super().summary(**kwargs)

    @CachedArtifactMixin.cache_result
    def details(self, **kwargs) -> dict:
        """Retrieve the CachedEndpoint Details.

        Returns:
            dict: A dictionary of details about the CachedEndpoint
        """
        return super().details(**kwargs)

    @CachedArtifactMixin.cache_result
    def health_check(self, **kwargs) -> dict:
        """Retrieve the CachedEndpoint Health Check.

        Returns:
            dict: A dictionary of health check details for the CachedEndpoint
        """
        return super().health_check(**kwargs)

    @CachedArtifactMixin.cache_result
    def workbench_meta(self) -> Union[str, None]:
        """Retrieve the Enumerated Model Type (REGRESSOR, CLASSIFER, etc).

        Returns:
            str: The Enumerated Model Type
        """
        return super().workbench_meta()

    @CachedArtifactMixin.cache_result
    def endpoint_metrics(self) -> Union[str, None]:
        """Retrieve the Endpoint Metrics

        Returns:
            str: The Endpoint Metrics
        """
        return super().endpoint_metrics()


if __name__ == "__main__":
    """Exercise the CachedEndpoint Class"""
    from pprint import pprint

    # Retrieve an existing Endpoint
    my_endpoint = CachedEndpoint("abalone-regression")
    pprint(my_endpoint.summary())
    pprint(my_endpoint.details())
    pprint(my_endpoint.health_check())
    print(my_endpoint.endpoint_metrics())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_endpoint._shutdown()
