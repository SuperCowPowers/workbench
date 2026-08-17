"""Tests for LocalModel training (no AWS required)

These run the real generated model script as a subprocess, so they're slower than
the rest of the local suite.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from workbench.local import LocalDataSource, LocalModel, ModelType, ModelFramework
from workbench.utils import job_tracker
from workbench.utils.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def feature_set():
    """A small regression feature set"""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({"id": range(n), "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df["y"] = 2 * df["a"] - df["b"] + rng.normal(0, 0.2, n)
    return LocalDataSource(df, name="reg_data").to_features("reg_features", id_column="id")


class TestTraining:
    def test_trains_and_records_outcome(self, feature_set):
        model = feature_set.to_model(
            "reg-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
        )

        assert model.training_state()["state"] == "completed"
        assert model.training_state()["returncode"] == 0
        assert model.get_status() == "ready"

        # The script wrote real artifacts and out-of-fold predictions
        assert len(model.oof_predictions()) == 100
        assert model.workbench_meta()["workbench_model_features"] == ["a", "b"]
        assert model.workbench_meta()["workbench_model_target"] == "y"

    def test_validation_ids_are_held_out(self, feature_set):
        model = feature_set.to_model(
            "holdout-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            validation_ids=list(range(90, 100)),
        )

        # Held-out rows are scored separately and kept out of the out-of-fold set
        assert len(model.validation_predictions()) == 10
        assert len(model.oof_predictions()) == 90

    def test_excluded_ids_never_reach_the_model(self, feature_set):
        model = feature_set.to_model(
            "excluded-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            exclude_ids=list(range(80, 100)),
        )
        assert len(model.oof_predictions()) == 80

    def test_script_is_kept_with_the_model(self, feature_set):
        model = feature_set.to_model(
            "kept-script",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
        )
        script = model.training_log()
        assert "Model training complete" in script

        # The generated script travels with the model, so the run is reproducible
        with open(f"{model.script_dir}/generated_model_script.py") as fp:
            source = fp.read()
        assert '"target": "y"' in source or "'target': 'y'" in source or '"y"' in source


class TestFailure:
    def test_failure_raises_and_records_state(self, feature_set):
        with pytest.raises(RuntimeError, match="Local training failed"):
            feature_set.to_model(
                "bad-model",
                model_type=ModelType.REGRESSOR,
                model_framework=ModelFramework.XGBOOST,
                target_column="not_a_column",
                feature_list=["a", "b"],
            )

        model = LocalModel("bad-model")
        assert model.training_state()["state"] == "failed"
        assert model.get_status() == "failed"
        assert model.training_state()["returncode"] != 0

        # The log is kept so the failure is diagnosable after the fact
        assert "not_a_column" in model.training_log()

    def test_target_required_for_supervised(self, feature_set):
        with pytest.raises(ValueError, match="target_column is required"):
            feature_set.to_model(
                "no-target",
                model_type=ModelType.REGRESSOR,
                model_framework=ModelFramework.XGBOOST,
            )


class TestRerun:
    def test_failed_retrain_does_not_serve_stale_predictions(self, feature_set):
        """A previous run's outputs must not survive into a failed one"""
        model = feature_set.to_model(
            "rerun-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
        )
        assert len(model.oof_predictions()) == 100

        with pytest.raises(RuntimeError):
            feature_set.to_model(
                "rerun-model",
                model_type=ModelType.REGRESSOR,
                model_framework=ModelFramework.XGBOOST,
                target_column="not_a_column",
                feature_list=["a", "b"],
            )

        assert LocalModel("rerun-model").oof_predictions().empty


class TestInterruptedRun:
    def test_dead_pid_reports_interrupted(self, feature_set):
        """A watcher that never recorded an outcome must not leave the model 'training'"""
        model = feature_set.to_model(
            "interrupted-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
        )

        # Simulate a session that exited mid-training: state left at training, pid gone.
        # Spawn and reap a child so the pid is known-dead rather than merely improbable
        # (a hardcoded high pid is unassignable on macOS but valid on Linux).
        dead = subprocess.Popen([sys.executable, "-c", ""])
        dead.wait()

        model._write_status(state="training", pid=dead.pid)
        assert LocalModel("interrupted-model").training_state()["state"] == "interrupted"

    def test_live_pid_still_reports_training(self, feature_set):
        model = feature_set.to_model(
            "live-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
        )

        model._write_status(state="training", pid=os.getpid())
        assert LocalModel("live-model").training_state()["state"] == "training"


class TestDetached:
    def test_wait_false_returns_immediately_then_finalizes(self, feature_set):
        job_tracker.drain_completed()

        model = feature_set.to_model(
            "async-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            wait=False,
        )

        # Returns while the child is still running, with a pid recorded for reattach
        assert model.training_state()["state"] == "training"
        assert model.training_state()["pid"] > 0

        rows = _wait_for_job()
        assert rows[0]["kind"] == "Local training"
        assert rows[0]["status"] == "COMPLETED"

        # A detached run leaves the same on-disk state a blocking one would
        fresh = LocalModel("async-model")
        assert fresh.training_state()["state"] == "completed"
        assert fresh.get_status() == "ready"
        assert len(fresh.oof_predictions()) == 100


def _wait_for_job(timeout: float = 180.0) -> list:
    """Poll the tracker until the training job reports in"""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = job_tracker.drain_completed()
        if rows:
            return rows
        time.sleep(0.2)
    raise AssertionError("training job never reported")
