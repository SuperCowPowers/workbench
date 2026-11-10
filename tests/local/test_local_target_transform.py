"""Target transforms end to end, through the real generated model script.

These run the template the way AWS does — training as a subprocess, then ``model_fn`` and
``predict_fn`` for inference — so they cover the forward/inverse placement inside the
template itself, which unit tests on the transform registry cannot reach.
"""

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


PERCENT_FLOOR = 0.1  # the reporting floor of the percent fixtures below


def _percent_frame():
    """A bounded percent target: mass at the low end, 8 rows read as an exact zero, top near 100.

    The shape `logit` is for. Transformed space is narrow here -- roughly [-3, 2] under logit
    and [0, 7.6] under arcsinh -- so an un-inverted prediction cannot be mistaken for an
    inverted one. The zeros sit at the lowest values, where an assay reports below its limit.
    """
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({"id": range(n), "a": rng.uniform(0, 1, n), "b": rng.normal(0, 1, n)})
    df["y"] = (100.0 * df["a"] ** 3).clip(upper=99.0)
    df.loc[df["a"].nsmallest(8).index, "y"] = 0.0
    return df


@pytest.fixture
def percent_feature_set():
    """The percent target as the assay reports it, zeros included."""
    return DataSource(_percent_frame(), name="percent_data").to_features("percent_features", id_column="id")


@pytest.fixture
def floored_percent_feature_set():
    """The same target with the zeros clamped up to the reporting floor.

    What a logit needs, done where a FeatureSet does it: the transform itself clamps nothing,
    so the clamp has to be visible in the data.
    """
    df = _percent_frame()
    df["y"] = df["y"].clip(lower=PERCENT_FLOOR)
    return DataSource(df, name="floored_data").to_features("floored_features", id_column="id")


def _train(feature_set, name, hyperparameters, model_type=ModelType.REGRESSOR):
    model = feature_set.to_model(
        name,
        model_type=model_type,
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


def test_logit_transform_inverts_inside_its_ceiling(floored_percent_feature_set):
    """A logit model carries no prediction_bounds, and its inverse still cannot leave (0, 100)."""
    model = _train(
        floored_percent_feature_set,
        "logit-model",
        {"target_transform": {"name": "logit", "ceiling": 100.0}, "n_folds": 3},
    )
    assert not model.hyperparameters().get("prediction_bounds"), "the ceiling is the only bound"

    oof = model.oof_predictions()
    assert len(oof) == 120
    assert oof["prediction"].max() > 10, "an un-inverted prediction would top out near 2"
    assert oof["prediction"].min() > 0.0, "the inverse is strictly inside the ceiling"
    assert oof["prediction"].max() < 100.0
    assert oof["prediction"].corr(oof["y"], method="spearman") > 0.9

    # The template round-trips the target through forward/inverse, so a transform that clamped
    # would hand back a rewritten truth column and score against it
    truth = floored_percent_feature_set.pull_dataframe().set_index("id")["y"]
    np.testing.assert_allclose(
        oof.set_index("id")["y"].sort_index(), truth.sort_index(), err_msg="saved truth must be untouched"
    )

    # predict_fn has to invert from the spec it reads back, with the same ceiling
    endpoint = model.to_endpoint()
    predictions = endpoint.inference(floored_percent_feature_set.pull_dataframe())
    assert predictions["prediction"].min() > 0.0
    assert predictions["prediction"].max() < 100.0
    assert predictions["prediction"].corr(predictions["y"], method="spearman") > 0.9


def test_arcsinh_transform_trains_through_exact_zeros(percent_feature_set):
    """The zeros go through arcsinh untouched, with nothing to configure for them."""
    model = _train(
        percent_feature_set,
        "arcsinh-model",
        {"target_transform": {"name": "arcsinh", "scale": 0.1}, "n_folds": 3},
    )

    oof = model.oof_predictions()
    assert (oof["y"] == 0.0).sum() == 8, "the zero rows trained rather than failing a domain check"
    assert oof["prediction"].max() > 20, "an un-inverted prediction would top out near 7.6"
    assert oof["prediction"].corr(oof["y"], method="spearman") > 0.9

    endpoint = model.to_endpoint()
    predictions = endpoint.inference(percent_feature_set.pull_dataframe())
    assert predictions["prediction"].max() > 20
    assert predictions["prediction"].corr(predictions["y"], method="spearman") > 0.9


def _logit_model(feature_set, name, ceiling):
    return feature_set.to_model(
        name,
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="y",
        feature_list=["a", "b"],
        hyperparameters={"target_transform": {"name": "logit", "ceiling": ceiling}, "n_folds": 3},
    )


def test_logit_above_its_ceiling_fails_at_creation(floored_percent_feature_set):
    """No training job runs: the target's range already says the ceiling cannot work."""
    with pytest.raises(ValueError, match=r"floored_features column 'y'") as failure:
        _logit_model(floored_percent_feature_set, "low-ceiling-model", 50.0)

    assert "needs target < ceiling" in str(failure.value)
    assert "cap those rows in the FeatureSet" in str(failure.value)


def test_logit_on_an_unfloored_target_fails_at_creation(percent_feature_set):
    """Sub-LLOQ rows are the FeatureSet's to clamp, and the error has to say so."""
    with pytest.raises(ValueError, match="needs target > 0") as failure:
        _logit_model(percent_feature_set, "unfloored-model", 100.0)

    assert "percent_features column 'y'" in str(failure.value)
    assert "drop them in the FeatureSet" in str(failure.value)
    assert "arcsinh" in str(failure.value), "and it names the transform with no domain limit"


def test_prediction_bounds_clip_uq_quantiles_but_not_the_spread(decades_feature_set):
    """Bounds reach the quantiles a UQ model emits, and stop at the spread."""
    bounded = _train(
        decades_feature_set,
        "uq-bounded-model",
        {"prediction_bounds": [0.0, 100.0], "n_folds": 3},
        model_type=ModelType.UQ_REGRESSOR,
    )
    unbounded = _train(
        decades_feature_set,
        "uq-unbounded-model",
        {"n_folds": 3},
        model_type=ModelType.UQ_REGRESSOR,
    )

    bounded_oof, unbounded_oof = bounded.oof_predictions(), unbounded.oof_predictions()
    quantiles = [c for c in bounded_oof.columns if c.startswith("q_")]
    assert quantiles, "a UQ model emits quantile columns"

    assert bounded_oof[quantiles].max().max() <= 100.0
    assert bounded_oof[quantiles].min().min() >= 0.0
    assert unbounded_oof[quantiles].max().max() > 100.0, "and the clip is what holds them there"
    np.testing.assert_allclose(
        bounded_oof["prediction_std"], unbounded_oof["prediction_std"], err_msg="the spread is not clipped"
    )


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


def test_out_of_domain_target_fails_at_creation():
    """A zero target under a zero offset fails before a job starts, not on NaNs."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(40), "a": rng.uniform(0, 1, 40), "b": rng.normal(0, 1, 40)})
    df["y"] = np.where(df.index < 5, 0.0, 10.0)
    feature_set = DataSource(df, name="zeros_data").to_features("zeros_features", id_column="id")

    with pytest.raises(ValueError, match=r"zeros_features column 'y'") as failure:
        feature_set.to_model(
            "zeros-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="y",
            feature_list=["a", "b"],
            hyperparameters={"target_transform": "log10", "n_folds": 3},
        )

    assert "needs target + offset > 0" in str(failure.value)
    assert "Raise the offset" in str(failure.value)
