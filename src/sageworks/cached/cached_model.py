"""CachedModel: Caches the method results for SageWorks Models"""

from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.model_core import ModelCore
from sageworks.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedModel(CachedArtifactMixin, ModelCore):
    """CachedModel: Caches the method results for SageWorks Models

    Note: Cached method values may lag underlying Model changes.

    Common Usage:
        ```python
        my_model = CachedModel(name)
        my_model.details()
        my_model.health_check()
        my_model.sageworks_meta()
        ```
    """

    def __init__(self, uuid: str):
        """CachedModel Initialization"""
        ModelCore.__init__(self, model_uuid=uuid, use_cached_meta=True)

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

    @CachedArtifactMixin.cache_result
    def sageworks_meta(self) -> Union[str, None]:
        """Retrieve the Enumerated Model Type (REGRESSOR, CLASSIFER, etc).

        Returns:
            str: The Enumerated Model Type
        """
        return super().sageworks_meta()

    @CachedArtifactMixin.cache_result
    def get_endpoint_inference_path(self) -> Union[str, None]:
        """Retrieve the Endpoint Inference Path.

        Returns:
            str: The Endpoint Inference Path
        """
        return super().get_endpoint_inference_path()

    @CachedArtifactMixin.cache_result
    def list_inference_runs(self) -> list[str]:
        """Retrieve the captured prediction results for this model

        Returns:
            list[str]: List of Inference Runs
        """
        return super().list_inference_runs()

    @CachedArtifactMixin.cache_result
    def get_inference_predictions(self, capture_uuid: str = "auto_inference") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: training_holdout)

        Returns:
            pd.DataFrame: DataFrame of the Captured Predictions (might be None)
        """
        return super().get_inference_predictions(capture_uuid=capture_uuid)


if __name__ == "__main__":
    """Exercise the CachedModel Class"""
    from pprint import pprint

    # Retrieve an existing Model
    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.health_check())
    print(my_model.get_endpoint_inference_path())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_model._shutdown()
