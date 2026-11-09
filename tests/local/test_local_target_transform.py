"""Target transforms end to end, through the real generated model script.

These run the template the way AWS does — training as a subprocess, then ``model_fn`` and
``predict_fn`` for inference — so they cover the forward/inverse placement inside the
template itself, which unit tests on the transform registry cannot reach.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workbench.local import DataSource, ModelFramework, ModelType
from workbench.utils.config_manager import ConfigManager

pytestmark = pytest.mark.medium

# The target spans five decades, so a prediction left in log10 space lands in [-2, 3] while an
# inverted one lands in the thousands. Any missing inverse is then unmissable.
TARGET_LOW, TARGET_HIGH = 0.01, 1000.0


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def decades_feature_set():
    """A regression feature set whose target spans five orders of magnitude."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({"id": range(n), "a": rng.uniform(0, 1, n), "b": rng.normal(0, 1, n)})
    exponent = np.log10(TARGET_LOW) + df["a"] * (np.log10(TARGET_HIGH) - np.log10(TARGET_LOW))
    df["y"] = 10**exponent + 0.01 * df["b"].abs()
    return DataSource(df, name="decades_data").to_features("decades_features", id_column="id")


def _train(feature_set, name, hyperparameters):
    model = feature_set.to_model(
        name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="y",
        feature_list=["a", "b"],
        hyperparameters=hyperparameters,
    )
    assert model.training_state()["returncode"] == 0, model.training_log()
    return model


def test_log10_transform_round_trips_through_training_and_inference(decades_feature_set):
    """Out-of-fold and endpoint predictions both come back on the target's own scale."""
    model = _train(
        decades_feature_set,
        "log10-model",
        {"target_transform": {"name": "log10", "offset": 0.001}, "n_folds": 3},
    )

    oof = model.oof_predictions()
    assert len(oof) == 120
    # Truth is stored untransformed, and predictions sit on the same scale
    assert oof["y"].max() > 100, "saved target should be original-scale"
    assert oof["prediction"].max() > 100, "an un-inverted prediction would top out near 3"
    assert oof["prediction"].min() > 0, "log10 inverts to strictly positive values"
    # Rank correlation, not Pearson: over five decades Pearson is dominated by the top one
    assert oof["prediction"].corr(oof["y"], method="spearman") > 0.9

    # The same inverse has to happen in predict_fn, via the spec read back from the model
    endpoint = model.to_endpoint()
    predictions = endpoint.inference(decades_feature_set.pull_dataframe())
    assert predictions["prediction"].max() > 100
    assert predictions["prediction"].min() > 0
    assert predictions["prediction"].corr(predictions["y"], method="spearman") > 0.9


def test_transformed_and_untransformed_models_agree_on_scale(decades_feature_set):
    """A transform changes accuracy, never the units predictions are reported in."""
    plain = _train(decades_feature_set, "plain-model", {"n_folds": 3})
    logged = _train(
        decades_feature_set,
        "logged-model",
        {"target_transform": "log10", "n_folds": 3},
    )

    plain_oof, logged_oof = plain.oof_predictions(), logged.oof_predictions()
    assert plain_oof["prediction"].median() > 1
    assert logged_oof["prediction"].median() > 1
    # Same order of magnitude: both are reporting on the original scale
    ratio = logged_oof["prediction"].median() / plain_oof["prediction"].median()
    assert 0.1 < ratio < 10


def test_prediction_bounds_clamp_a_linear_target(decades_feature_set):
    """The bounds-only case: no transform, just a physically valid range."""
    model = _train(
        decades_feature_set,
        "bounded-model",
        {"prediction_bounds": [0.0, 100.0], "n_folds": 3},
    )

    oof = model.oof_predictions()
    assert oof["prediction"].min() >= 0.0
    assert oof["prediction"].max() <= 100.0
    assert oof["y"].max() > 100.0, "the target itself is not clamped, only predictions"

    endpoint = model.to_endpoint()
    predictions = endpoint.inference(decades_feature_set.pull_dataframe())
    assert predictions["prediction"].min() >= 0.0
    assert predictions["prediction"].max() <= 100.0


def test_unsupported_framework_is_rejected_at_creation(decades_feature_set):
    """A template that would drop the spec must fail before any training runs."""
    with pytest.raises(ValueError, match="does not support target_transform"):
        decades_feature_set.to_model(
            "sklearn-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.SKLEARN,
            target_column="y",
            feature_list=["a", "b"],
            model_class="RandomForestRegressor",
            model_import_str="from sklearn.ensemble import RandomForestRegressor",
            hyperparameters={"target_transform": "log10"},
        )


def test_classifier_is_rejected_at_creation(decades_feature_set):
    with pytest.raises(ValueError, match="applies to regression targets only"):
        decades_feature_set.to_model(
            "classifier-model",
            model_type=ModelType.CLASSIFIER,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            hyperparameters={"target_transform": "log10"},
        )


def test_out_of_domain_target_fails_the_training_job():
    """A zero target under a zero offset has to fail loudly, not train on NaNs."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(40), "a": rng.uniform(0, 1, 40), "b": rng.normal(0, 1, 40)})
    df["y"] = np.where(df.index < 5, 0.0, 10.0)
    feature_set = DataSource(df, name="zeros_data").to_features("zeros_features", id_column="id")

    with pytest.raises(RuntimeError, match="Local training failed") as failure:
        feature_set.to_model(
            "zeros-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            hyperparameters={"target_transform": "log10", "n_folds": 3},
        )

    log_path = Path(str(failure.value).rsplit("see ", 1)[-1])
    assert "needs target + offset > 0" in log_path.read_text()
