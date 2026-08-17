"""Tests for LocalEndpoint inference (no AWS required)

These load an XGBoost model in-process. If the worker already imported torch, the
two OpenMP runtimes segfault the interpreter, so the module skips rather than taking
the worker down -- under `-n auto --dist=loadfile` a worker may have run a torch test
file first. Remove this when in-process loading is isolated.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from workbench.local import LocalDataSource, LocalEndpoint, LocalModel, ModelType, ModelFramework
from workbench.utils.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def skip_when_torch_loaded():
    """Checked per test, not at import: the worker may load torch after collection"""
    if "torch" in sys.modules:
        pytest.skip("torch is already imported in this worker; loading an XGBoost model here would segfault")


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def trained_model():
    """A trained local regression model"""
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({"id": range(n), "mw": rng.normal(300, 50, n), "logp": rng.normal(2.5, 1, n)})
    df["sol"] = -0.01 * df["mw"] - 0.5 * df["logp"] + rng.normal(0, 0.3, n)

    fs = LocalDataSource(df, name="e_data").to_features("e_features", id_column="id")
    return fs.to_model(
        "e-model",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="sol",
        feature_list=["mw", "logp"],
    )


class TestInferenceBundle:
    def test_training_writes_the_inference_bundle(self, trained_model):
        """The bundling step the SageMaker harness does, so the model dir is servable"""
        artifacts = os.listdir(trained_model.model_dir)
        assert "inference-metadata.json" in artifacts
        assert "generated_model_script.py" in artifacts


class TestEndpoint:
    def test_inference_returns_predictions(self, trained_model):
        endpoint = trained_model.to_endpoint()
        eval_df = LocalDataSource("e_data").pull_dataframe().head(10)

        predictions = endpoint.inference(eval_df)
        assert len(predictions) == 10
        assert "prediction" in predictions.columns

    def test_uq_columns_present(self, trained_model):
        """A UQ regressor serves quantiles, same as a deployed endpoint"""
        endpoint = trained_model.to_endpoint()
        predictions = endpoint.inference(LocalDataSource("e_data").pull_dataframe().head(5))

        assert "prediction_std" in predictions.columns
        assert {"q_025", "q_50", "q_975"}.issubset(set(predictions.columns))

    def test_default_endpoint_name(self, trained_model):
        assert trained_model.to_endpoint().name == "e-model-end"

    def test_explicit_endpoint_name(self, trained_model):
        assert trained_model.to_endpoint("custom-end").name == "custom-end"

    def test_model_bundle_loaded_once(self, trained_model):
        endpoint = trained_model.to_endpoint()
        eval_df = LocalDataSource("e_data").pull_dataframe().head(3)

        endpoint.inference(eval_df)
        first = endpoint._model_bundle
        endpoint.inference(eval_df)
        assert endpoint._model_bundle is first

    def test_untrained_model_cannot_be_served(self, trained_model):
        """A model that never trained has nothing to load"""
        with pytest.raises(ValueError, match="has not trained successfully"):
            LocalEndpoint.from_model(LocalModel("never-trained"))


class TestCaptures:
    def test_capture_and_recall(self, trained_model):
        endpoint = trained_model.to_endpoint()
        eval_df = LocalDataSource("e_data").pull_dataframe().head(8)

        endpoint.inference(eval_df, capture_name="auto_inference")
        assert endpoint.list_captures() == ["auto_inference"]

        recalled = endpoint.get_inference_predictions("auto_inference")
        assert len(recalled) == 8

        # A fresh handle reads the same captures off disk
        assert LocalEndpoint(endpoint.name).list_captures() == ["auto_inference"]

    def test_missing_capture_returns_none(self, trained_model):
        endpoint = trained_model.to_endpoint()
        assert endpoint.get_inference_predictions("nope") is None

    def test_inference_without_capture_stores_nothing(self, trained_model):
        endpoint = trained_model.to_endpoint()
        endpoint.inference(LocalDataSource("e_data").pull_dataframe().head(3))
        assert endpoint.list_captures() == []

    def test_sm_model_dir_is_restored(self, trained_model):
        """predict_fn resolves files through SM_MODEL_DIR, so it must not leak"""
        os.environ["SM_MODEL_DIR"] = "/sentinel"
        try:
            endpoint = trained_model.to_endpoint()
            endpoint.inference(LocalDataSource("e_data").pull_dataframe().head(3))
            assert os.environ["SM_MODEL_DIR"] == "/sentinel"
        finally:
            os.environ.pop("SM_MODEL_DIR", None)
