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
from workbench.local.local_artifact import LocalArtifact
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
        """An endpoint takes its model's name, same as the AWS side"""
        assert trained_model.to_endpoint().name == "e-model"

    def test_explicit_endpoint_name(self, trained_model):
        assert trained_model.to_endpoint("custom-serving").name == "custom-serving"

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


class TestModelDelete:
    def test_delete_takes_the_endpoint_with_it(self, trained_model):
        """An endpoint has no meaning without its model, so it comes down too"""
        endpoint = trained_model.to_endpoint()
        trained_model.delete()

        assert not trained_model.exists()
        assert not LocalEndpoint(endpoint.name).exists()

    def test_custom_named_endpoint_is_found(self, trained_model):
        """The cascade matches on model_name, not on the naming convention"""
        endpoint = trained_model.to_endpoint("custom-serving")
        trained_model.delete()

        assert not LocalEndpoint(endpoint.name).exists()

    def test_other_models_endpoints_are_untouched(self, trained_model):
        """Only the endpoints serving this model come down"""
        endpoint = trained_model.to_endpoint()
        bystander = LocalEndpoint("someone-elses")
        bystander._init_storage(input_name="other-model")
        bystander.upsert_workbench_meta({"model_name": "other-model"})

        trained_model.delete()

        assert not LocalEndpoint(endpoint.name).exists()
        assert LocalEndpoint("someone-elses").exists()

    def test_orphaned_endpoint_reports_the_missing_model(self, trained_model):
        """An endpoint orphaned some other way fails readably rather than on an open()"""
        endpoint = trained_model.to_endpoint()
        LocalArtifact.delete(trained_model)  # Skip the cascade to orphan the endpoint

        with pytest.raises(FileNotFoundError, match="No model artifacts"):
            endpoint.inference(pd.DataFrame({"mw": [300.0], "logp": [2.0]}))


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


class TestInferenceRuns:
    """The model reaches its runs the way an AWS Model does, so a script that walks
    inference runs works against either."""

    def test_training_run_is_listed(self, trained_model):
        assert trained_model.list_inference_runs() == ["full_cross_fold"]

    def test_endpoint_captures_join_the_list(self, trained_model):
        endpoint = trained_model.to_endpoint()
        endpoint.inference(LocalDataSource("e_data").pull_dataframe().head(5), capture_name="holdout")

        assert trained_model.list_inference_runs() == ["full_cross_fold", "holdout"]

    def test_default_prefers_cross_fold(self, trained_model):
        trained_model.to_endpoint().inference(
            LocalDataSource("e_data").pull_dataframe().head(5), capture_name="holdout"
        )
        assert trained_model.default_inference_run() == "full_cross_fold"

    def test_predictions_from_the_training_run(self, trained_model):
        predictions = trained_model.get_inference_predictions("full_cross_fold")
        assert predictions.equals(trained_model.oof_predictions())

    def test_predictions_from_an_endpoint_capture(self, trained_model):
        trained_model.to_endpoint().inference(
            LocalDataSource("e_data").pull_dataframe().head(5), capture_name="holdout"
        )
        assert len(trained_model.get_inference_predictions("holdout")) == 5

    def test_metrics_are_computed_from_predictions(self, trained_model):
        metrics = trained_model.get_inference_metrics("full_cross_fold")
        assert len(metrics) == 1
        assert {"rmse", "mae", "r2"}.issubset(set(metrics.columns))

    def test_unknown_run_returns_none(self, trained_model):
        assert trained_model.get_inference_predictions("nope") is None
        assert trained_model.get_inference_metrics("nope") is None

    def test_multi_task_scores_the_primary_target(self, trained_model, monkeypatch):
        """A multi-task model is scored on its first target, over the rows it covers"""
        predictions = trained_model.oof_predictions()
        predictions["other"] = np.nan
        predictions.loc[: len(predictions) // 2, "sol"] = np.nan
        monkeypatch.setattr(type(trained_model), "oof_predictions", lambda self: predictions)
        trained_model.upsert_workbench_meta({"workbench_model_target": ["sol", "other"]})

        metrics = trained_model.get_inference_metrics("full_cross_fold")
        assert metrics["support"].iloc[0] == predictions["sol"].notna().sum()

    def test_untrained_model_has_no_runs(self):
        model = LocalModel("never-trained")
        assert model.list_inference_runs() == []
        assert model.default_inference_run() is None
        assert model.get_inference_predictions() is None

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
