"""CachedModel: Caches the method results for Workbench Models"""

from typing import Union
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.model_core import ModelCore
from workbench.core.artifacts.cached_artifact_mixin import CachedArtifactMixin


class CachedModel(CachedArtifactMixin, ModelCore):
    """CachedModel: Caches the method results for Workbench Models

    Note: Cached method values may lag underlying Model changes.

    Common Usage:
        ```python
        my_model = CachedModel(name)
        my_model.details()
        my_model.health_check()
        my_model.workbench_meta()
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
    def workbench_meta(self) -> Union[str, None]:
        """Retrieve the Enumerated Model Type (REGRESSOR, CLASSIFER, etc).

        Returns:
            str: The Enumerated Model Type
        """
        return super().workbench_meta()

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
    def get_inference_metrics(self, capture_uuid: str = "latest") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: latest)

        Returns:
            pd.DataFrame: DataFrame of the Captured Metrics (might be None)
        """
        return super().get_inference_metrics(capture_uuid=capture_uuid)

    @CachedArtifactMixin.cache_result
    def get_inference_predictions(self, capture_uuid: str = "auto_inference") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: training_holdout)

        Returns:
            pd.DataFrame: DataFrame of the Captured Predictions (might be None)
        """
        # Note: This method can generate larger dataframes, so we'll sample if needed
        df = super().get_inference_predictions(capture_uuid=capture_uuid)
        if df is not None and len(df) > 5000:
            self.log.warning(f"{self.uuid}:{capture_uuid} Sampling Inference Predictions to 5000 rows")
            return df.sample(5000)
        return df

    @CachedArtifactMixin.cache_result
    def confusion_matrix(self, capture_uuid: str = "latest") -> Union[pd.DataFrame, None]:
        """Retrieve the confusion matrix for the model

        Args:
            capture_uuid (str, optional): Specific capture_uuid (default: latest)

        Returns:
            pd.DataFrame: DataFrame of the Confusion Matrix (might be None)
        """
        return super().confusion_matrix(capture_uuid=capture_uuid)


if __name__ == "__main__":
    """Exercise the CachedModel Class"""
    from pprint import pprint

    # Retrieve an existing Model
    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.list_inference_runs())
    print(my_model.get_inference_metrics())
    print(my_model.get_inference_predictions())

    # Shutdown the ThreadPoolExecutor (note: users should NOT call this)
    my_model._shutdown()
