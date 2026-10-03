"""Tests for the TabICL fit / save / load / probe helpers.

These fit real (tiny) TabICL models, so they need the ``modeling`` extra and the TabICL
checkpoint (local cache, Workbench bucket, or Hugging Face).
"""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabicl")

# Workbench Imports
from workbench.endpoints.tabicl_utils import MODEL_FILE, REDUCER_FILE, load_tabicl, predict_with_std  # noqa: E402
from workbench.training.tabicl_core import (  # noqa: E402
    fit_tabicl,
    probe_inference_memory,
    save_tabicl,
    top_variance_selector,
)

pytestmark = pytest.mark.medium

HYPERPARAMETERS = {"n_estimators": 2, "batch_size": 1, "seed": 42, "pca_components": None}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] * 2 + X["f1"] + rng.normal(scale=0.2, size=len(X))
    return X, y


@pytest.fixture(scope="module")
def served(data):
    X, y = data
    return fit_tabicl(HYPERPARAMETERS, X.iloc[:160], y.iloc[:160], cache=True)


def test_predict_with_std_returns_a_prediction_and_a_positive_spread(data, served):
    X, y = data
    model, reducer = served
    prediction, std = predict_with_std(model, X.iloc[160:], reducer)

    assert reducer is None
    assert prediction.shape == std.shape == (40,)
    assert (std > 0).all()
    assert np.abs(prediction - y.iloc[160:]).mean() < y.std()  # learned something from context


def test_cached_and_uncached_fits_agree(data, served):
    X, y = data
    uncached, _ = fit_tabicl(HYPERPARAMETERS, X.iloc[:160], y.iloc[:160], cache=False)
    np.testing.assert_allclose(
        predict_with_std(uncached, X.iloc[160:])[0], predict_with_std(served[0], X.iloc[160:])[0], atol=1e-3
    )


def test_pca_components_puts_a_reducer_in_front(data, tmp_path):
    X, y = data
    model, reducer = fit_tabicl({**HYPERPARAMETERS, "pca_components": 3}, X.iloc[:160], y.iloc[:160], cache=True)
    assert reducer is not None and model.n_features_in_ == 3

    save_tabicl(model, reducer, str(tmp_path))
    assert (tmp_path / REDUCER_FILE).exists()
    loaded_model, loaded_reducer = load_tabicl(str(tmp_path), device="cpu")
    np.testing.assert_allclose(
        predict_with_std(loaded_model, X.iloc[160:], loaded_reducer)[0],
        predict_with_std(model, X.iloc[160:], reducer)[0],
        atol=1e-4,
    )


def test_top_variance_selector_ranks_only_expanded_columns():
    """The n highest-variance expanded columns are kept, every other column passes through, in order"""
    rng = np.random.default_rng(0)
    scales = {"emb_0": 0.1, "emb_1": 5.0, "emb_2": 1.0, "emb_3": 3.0}
    X = pd.DataFrame({name: rng.normal(scale=s, size=100) for name, s in scales.items()})
    X.insert(1, "readout", rng.normal(scale=0.01, size=100))  # lowest variance, but not ranked

    selector = top_variance_selector(X, list(scales), 2)
    np.testing.assert_array_equal(selector.transform(X), X[["readout", "emb_1", "emb_3"]].to_numpy())
    assert top_variance_selector(X, list(scales), 10).transform(X).shape == X.shape  # n past the count keeps all


def test_top_variance_features_puts_a_selector_in_front(data, tmp_path):
    X, y = data
    hyperparameters = {**HYPERPARAMETERS, "top_variance_features": 2}
    expanded = ["f2", "f3", "f4", "f5"]
    model, reducer = fit_tabicl(hyperparameters, X.iloc[:160], y.iloc[:160], cache=True, expanded_columns=expanded)
    assert model.n_features_in_ == 4  # f0, f1 kept + 2 of the 4 expanded columns

    save_tabicl(model, reducer, str(tmp_path))
    loaded_model, loaded_reducer = load_tabicl(str(tmp_path), device="cpu")
    np.testing.assert_allclose(
        predict_with_std(loaded_model, X.iloc[160:], loaded_reducer)[0],
        predict_with_std(model, X.iloc[160:], reducer)[0],
        atol=1e-4,
    )


def test_save_and_load_round_trip(data, served, tmp_path):
    X, _ = data
    model, reducer = served
    save_tabicl(model, reducer, str(tmp_path))

    assert (tmp_path / MODEL_FILE).exists() and not (tmp_path / REDUCER_FILE).exists()
    loaded, _ = load_tabicl(str(tmp_path), device="cpu")
    np.testing.assert_allclose(
        predict_with_std(loaded, X.iloc[160:])[0], predict_with_std(model, X.iloc[160:])[0], atol=1e-4
    )


def test_a_gpu_fit_artifact_loads_on_a_cpu_host(data, served, tmp_path):
    """TabICL restores the estimator's `device` while unpickling, so the artifact must
    record "cpu" whatever device the model was fit on."""
    model, reducer = served
    fit_device = model.device
    model.set_params(device="cuda")  # what a GPU training job's estimator carries
    try:
        save_tabicl(model, reducer, str(tmp_path))
        assert model.device == "cuda"  # the fitted estimator is left as it was
    finally:
        model.set_params(device=fit_device)

    loaded, _ = load_tabicl(str(tmp_path), device="cpu")
    assert loaded.device == "cpu"


def test_probe_reports_peak_memory(data, served, tmp_path):
    X, _ = data
    save_tabicl(*served, str(tmp_path))
    assert 0.1 < probe_inference_memory(str(tmp_path), X) < 16


def test_probe_surfaces_why_a_model_fails_to_load(data, tmp_path):
    X, _ = data
    with open(os.path.join(tmp_path, MODEL_FILE), "wb") as f:
        f.write(b"not a model")
    with pytest.raises(RuntimeError, match="failed to load and predict on CPU"):
        probe_inference_memory(str(tmp_path), X)
