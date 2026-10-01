"""Dual-use TabICL helpers — shared by training and endpoint inference.

Lives on the endpoint import surface (per the :mod:`workbench.endpoints` contract)
because the serving ``predict_fn`` reaches the model the same way training does.
Top-level deps are numpy + joblib + tabicl, which ship in the ``pytorch_chem`` images.
"""

import os

import joblib
import numpy as np
from tabicl import TabICLRegressor

MODEL_FILE = "tabicl_model.pkl"
REDUCER_FILE = "tabicl_reducer.joblib"

# One standard deviation either side of the median, for a normal predictive distribution
STD_QUANTILES = [0.16, 0.84]


def tabicl_device() -> str:
    """CUDA when present, else CPU.

    Apple's MPS is skipped: it draws on system memory, which TabICL's in-context pass
    can exhaust.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tabicl(model_dir: str, device: str | None = None) -> tuple:
    """Load the served TabICL model and its optional feature reducer.

    The device is chosen where the model is loaded, not where it was fit, so a model
    fit on a GPU serves from a CPU endpoint.

    Args:
        model_dir: directory holding the saved model artifacts.
        device: torch device; None picks via :func:`tabicl_device`.

    Returns:
        tuple: (model, reducer) — reducer is None when the model takes raw features.
    """
    model = TabICLRegressor.load(os.path.join(model_dir, MODEL_FILE), device=device or tabicl_device())
    reducer_path = os.path.join(model_dir, REDUCER_FILE)
    reducer = joblib.load(reducer_path) if os.path.exists(reducer_path) else None
    return model, reducer


def predict_with_std(model, X, reducer=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean prediction and a std-like spread from TabICL's predictive distribution.

    The spread is half the distance between the 16th and 84th percentiles: the model's
    own per-row uncertainty, used as ``prediction_std`` by the UQModel.

    Args:
        model: a fitted TabICLRegressor.
        X: feature frame or array, in the model's training column order.
        reducer: fitted feature reducer, applied before the model when present.

    Returns:
        tuple: (prediction, prediction_std), each of shape (n_rows,).
    """
    if reducer is not None:
        X = reducer.transform(X)
    out = model.predict(X, output_type=["mean", "quantiles"], alphas=STD_QUANTILES)
    lower, upper = out["quantiles"][:, 0], out["quantiles"][:, 1]
    return np.asarray(out["mean"]), (upper - lower) / 2
