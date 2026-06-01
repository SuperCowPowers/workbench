"""Tests for model-training metrics metadata serialization."""

import importlib.util
import sys
import types
from pathlib import Path


class FakeMetricsFrame:
    """Small DataFrame stand-in for regression metrics serialization tests."""

    columns = ["metric_name", "value", "timestamp"]

    def __getitem__(self, key):
        data = {
            "metric_name": ["MAE", "RMSE", "R2"],
            "value": [0.77, 1.28, 0.84],
            "timestamp": ["unused", "unused", "unused"],
        }
        return data[key]


def _load_model_core_module():
    module_names = [
        "botocore",
        "botocore.exceptions",
        "awswrangler",
        "awswrangler.exceptions",
        "sagemaker",
        "sagemaker.core",
        "sagemaker.core.resources",
        "workbench",
        "workbench.core",
        "workbench.core.artifacts",
        "workbench.core.artifacts.artifact",
        "workbench.utils",
        "workbench.utils.aws_utils",
        "workbench.utils.metrics_utils",
        "workbench.utils.s3_utils",
        "workbench.utils.shap_utils",
        "workbench.utils.deprecated_utils",
        "workbench.utils.model_utils",
    ]
    originals = {name: sys.modules.get(name) for name in module_names}

    class FakeDataFrame:
        def __init__(self, data=None):
            self.data = data
            self.from_dict_input = None

        @classmethod
        def from_dict(cls, data):
            instance = cls()
            instance.from_dict_input = data
            return instance

    class FakeClientError(Exception):
        pass

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for attr_name, value in attrs.items():
            setattr(mod, attr_name, value)
        sys.modules[name] = mod
        return mod

    try:
        pandas = module("pandas", DataFrame=FakeDataFrame)
        exceptions = module("botocore.exceptions", ClientError=FakeClientError)
        module("botocore", exceptions=exceptions)
        wr_exceptions = module("awswrangler.exceptions", NoFilesFound=Exception)
        module("awswrangler", exceptions=wr_exceptions)
        module("sagemaker")
        module("sagemaker.core")
        module("sagemaker.core.resources", TrainingJob=object, ModelPackageGroup=object)
        module("workbench")
        module("workbench.core")
        module("workbench.core.artifacts")
        module("workbench.core.artifacts.artifact", Artifact=object)
        module("workbench.utils")
        module("workbench.utils.aws_utils", newest_path=lambda *args, **kwargs: None, pull_s3_data=lambda *args: None)
        module(
            "workbench.utils.metrics_utils",
            reorder_cm_df=lambda *args, **kwargs: None,
            reorder_metrics_df=lambda *args, **kwargs: None,
        )
        module("workbench.utils.s3_utils", compute_s3_object_hash=lambda *args, **kwargs: None)
        module(
            "workbench.utils.shap_utils",
            get_shap_importance=lambda *args, **kwargs: None,
            get_shap_values=lambda *args, **kwargs: None,
            get_shap_feature_values=lambda *args, **kwargs: None,
        )
        module("workbench.utils.deprecated_utils", deprecated=lambda *args, **kwargs: (lambda func: func))
        module(
            "workbench.utils.model_utils",
            published_proximity_model=lambda *args, **kwargs: None,
            get_model_hyperparameters=lambda *args, **kwargs: None,
        )

        module_path = Path(__file__).parents[2] / "src" / "workbench" / "core" / "artifacts" / "model_core.py"
        spec = importlib.util.spec_from_file_location("model_core_under_test", module_path)
        model_core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_core)
        model_core.pd = pandas
        return model_core
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_regression_training_metrics_are_flat_metadata():
    model_core = _load_model_core_module()

    assert model_core.ModelCore._regression_training_metrics_metadata(FakeMetricsFrame()) == {
        "MAE": 0.77,
        "RMSE": 1.28,
        "R2": 0.84,
    }


def test_flat_training_metrics_metadata_reads_as_single_row_dataframe():
    model_core = _load_model_core_module()
    flat_metrics = {"MAE": 0.77, "RMSE": 1.28}

    metrics_df = model_core.ModelCore._training_metrics_from_metadata(flat_metrics)

    assert metrics_df.data == [flat_metrics]
    assert metrics_df.from_dict_input is None


def test_nested_training_metrics_metadata_keeps_legacy_reader():
    model_core = _load_model_core_module()
    legacy_metrics = {"MAE": {"0": 0.77}, "RMSE": {"0": 1.28}}

    metrics_df = model_core.ModelCore._training_metrics_from_metadata(legacy_metrics)

    assert metrics_df.data is None
    assert metrics_df.from_dict_input == legacy_metrics
