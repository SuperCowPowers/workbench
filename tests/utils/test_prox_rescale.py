"""Tests for moving a proximity's target columns between the original scale and a model's space."""

import numpy as np
import pandas as pd
import pytest

from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity
from workbench.utils.prox_utils import rescale_proximity_targets

FEATURES = ["f1", "f2"]


@pytest.fixture
def prox():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.normal(size=30) for f in FEATURES})
    df["id"] = [f"r{i}" for i in range(30)]
    df["cmax"] = np.power(10.0, rng.uniform(0, 4, 30))
    df["logd"] = rng.normal(size=30)
    df.loc[:4, "cmax"] = np.nan
    return FeatureSpaceProximity(df, id_column="id", features=FEATURES, target=["cmax", "logd"])


def test_untransformed_model_gets_the_same_proximity(prox):
    assert rescale_proximity_targets(prox, {}, to_model_space=True) is prox
    assert rescale_proximity_targets(prox, {"target_transform": None}, to_model_space=False) is prox


def test_forward_moves_only_the_transformed_target(prox):
    hyperparameters = {"target_transform": {"cmax": {"name": "log10"}}}
    model_space = rescale_proximity_targets(prox, hyperparameters, to_model_space=True)

    np.testing.assert_allclose(model_space.df["cmax"], np.log10(prox.df["cmax"]))
    np.testing.assert_allclose(model_space.df["logd"], prox.df["logd"])
    assert model_space.df["cmax"].isna().sum() == 5, "unlabeled rows stay unlabeled"
    assert prox.df["cmax"].max() > 100, "the source proximity is left alone"


def test_inverse_undoes_forward(prox):
    hyperparameters = {"target_transform": {"cmax": {"name": "log10"}}}
    model_space = rescale_proximity_targets(prox, hyperparameters, to_model_space=True)
    original = rescale_proximity_targets(model_space, hyperparameters, to_model_space=False)
    np.testing.assert_allclose(original.df["cmax"], prox.df["cmax"], rtol=1e-9)


def test_neighbors_carry_the_converted_targets(prox):
    hyperparameters = {"target_transform": "log10"}
    single = FeatureSpaceProximity(prox.df.dropna(subset=["cmax"]), id_column="id", features=FEATURES, target="cmax")
    model_space = rescale_proximity_targets(single, hyperparameters, to_model_space=True)
    neighbors = model_space.neighbors("r10", n_neighbors=3, include_self=False)
    assert neighbors["cmax"].max() < 5.0, "neighbor targets read in log units"
