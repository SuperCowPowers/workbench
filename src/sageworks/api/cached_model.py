"""CachedModel: Caches the method results for SageWorks Models"""

# SageWorks Imports
from sageworks.core.artifacts.model_core import ModelCore
from sageworks.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedModel(CachedArtifactMixin, ModelCore):
    """CachedModel: SageWorks CachedModel API Class.

    Common Usage:
        ```python
        my_model = CachedModel(name)
        my_model.details()
        my_model.to_endpoint()
        ```
    """
    def __init__(self, uuid: str):
        super().__init__(uuid)  # Use super() to call both mixin and base class initializers

    @CachedArtifactMixin.cache_result
    def summary(self, **kwargs) -> dict:
        """Retrieve the CachedModel Details.

        Returns:
            dict: A dictionary of details about the CachedModel
        """
        return super().summary(**kwargs)

    @CachedArtifactMixin.cache_result
    def details(self, **kwargs) -> dict:
        """Retrieve the CachedModel Details.

        Returns:
            dict: A dictionary of details about the CachedModel
        """
        return super().details(**kwargs)

    @CachedArtifactMixin.cache_result
    def health_check(self, **kwargs) -> dict:
        """Retrieve the CachedModel Health Check.

        Returns:
            dict: A dictionary of health check details for the CachedModel
        """
        return super().health_check(**kwargs)


if __name__ == "__main__":
    """Exercise the CachedModel Class"""
    from pprint import pprint

    # Retrieve an existing Model
    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.health_check())
    my_model.close()
