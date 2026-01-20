"""CachedModel: Caches the method results for Workbench Models"""

from typing import Union
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.model_core import ModelCore, ModelType
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

    def __init__(self, name: str):
        """CachedModel Initialization"""
        ModelCore.__init__(self, model_name=name, use_cached_meta=True)

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
    def get_inference_metrics(self, capture_name: str = "auto") -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_name (str, optional): Specific capture_name (default: auto)

        Returns:
            pd.DataFrame: DataFrame of the Captured Metrics (might be None)
        """
        return super().get_inference_metrics(capture_name=capture_name)

    @CachedArtifactMixin.cache_result
    def get_inference_predictions(
        self, capture_name: str = "full_cross_fold", limit: int = 5000
    ) -> Union[pd.DataFrame, None]:
        """Retrieve the captured prediction results for this model

        Args:
            capture_name (str, optional): Specific capture_name (default: auto_inference)
            limit (int, optional): Maximum rows to return (default: 1000)

        Returns:
            pd.DataFrame: DataFrame of the Captured Predictions (might be None)
        """
        df = super().get_inference_predictions(capture_name=capture_name)
        if df is None:
            return None

        # Compute residual and do smart sampling based on model type
        is_regressor = self.model_type in [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]
        is_classifier = self.model_type == ModelType.CLASSIFIER

        if is_regressor:
            target = self.target()
            if target and "prediction" in df.columns and target in df.columns:
                df["residual"] = abs(df["prediction"] - df[target])

        elif is_classifier:
            target = self.target()
            class_labels = self.class_labels()
            if target and "prediction" in df.columns and target in df.columns and class_labels:
                # Create a mapping from label to ordinal index
                label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
                # Compute residual as distance between predicted and actual class
                df["residual"] = abs(
                    df["prediction"].map(label_to_idx).fillna(-1) - df[target].map(label_to_idx).fillna(-1)
                )

        # Smart sampling: half high-residual rows, half random from the rest
        if "residual" in df.columns and len(df) > limit:
            half_limit = limit // 2
            self.log.warning(
                f"{self.name}:{capture_name} Sampling {limit} rows (top {half_limit} residuals + {half_limit} random)"
            )
            top_residuals = df.nlargest(half_limit, "residual")
            remaining = df.drop(top_residuals.index)
            random_sample = remaining.sample(min(half_limit, len(remaining)))
            return pd.concat([top_residuals, random_sample]).reset_index(drop=True)

        # Fallback: just limit rows if no residual computed
        if len(df) > limit:
            self.log.warning(f"{self.name}:{capture_name} Sampling to {limit} rows")
            return df.sample(limit)

        return df

    @CachedArtifactMixin.cache_result
    def confusion_matrix(self, capture_name: str = "auto") -> Union[pd.DataFrame, None]:
        """Retrieve the confusion matrix for the model

        Args:
            capture_name (str, optional): Specific capture_name (default: auto)

        Returns:
            pd.DataFrame: DataFrame of the Confusion Matrix (might be None)
        """
        return super().confusion_matrix(capture_name=capture_name)


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
