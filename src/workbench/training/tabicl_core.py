"""TabICL training primitives shared by the tabicl template and local experiments.

TabICL is an in-context model: ``fit`` stores the training rows (optionally caching their
transformer projections) and every prediction conditions on them. Nothing is trained, so
the concerns here are caching those rows and how much memory serving needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from workbench.endpoints.tabicl_utils import MODEL_FILE, REDUCER_FILE, tabicl_device

# Rows the memory probe predicts: a typical small serving request
PROBE_ROWS = 10


def fit_tabicl(hyperparameters: dict, X: pd.DataFrame, y, *, cache: bool) -> tuple:
    """Fit a TabICL regressor, with the optional feature reducer in front.

    Args:
        hyperparameters: the template's resolved hyperparameters.
        X: training features.
        y: training target.
        cache: cache the training rows' transformer projections. True for a model that
            answers many requests (each then costs only its own rows); False for a
            throwaway model that predicts once, such as a fold model.

    Returns:
        tuple: (model, reducer) — reducer is None unless ``pca_components`` is set.
    """
    from tabicl import TabICLRegressor

    from workbench.training.foundation_models import resolve_foundation_checkpoint

    reducer = None
    if hyperparameters.get("pca_components"):
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        reducer = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            PCA(n_components=hyperparameters["pca_components"], random_state=hyperparameters["seed"]),
        ).fit(X)
        X = reducer.transform(X)

    model = TabICLRegressor(
        n_estimators=hyperparameters["n_estimators"],
        batch_size=hyperparameters["batch_size"],
        kv_cache="kv" if cache else False,
        random_state=hyperparameters["seed"],
        device=tabicl_device(),
        model_path=str(resolve_foundation_checkpoint("tabicl")),
        allow_auto_download=False,
    ).fit(X, y)
    return model, reducer


def save_tabicl(model, reducer, model_dir: str) -> None:
    """Save the served model as a portable, self-contained artifact.

    Pretrained weights travel in the artifact, so an endpoint never reaches for Hugging
    Face. TabICL restores the estimator's ``device`` parameter while unpickling, so the
    artifact records "cpu": it then loads on any host, and ``load_tabicl`` moves it to
    the device that host has.
    """
    import joblib

    fit_device = model.device
    model.set_params(device="cpu")
    try:
        model.save(os.path.join(model_dir, MODEL_FILE), save_model_weights=True)
    finally:
        model.set_params(device=fit_device)
    if reducer is not None:
        joblib.dump(reducer, os.path.join(model_dir, REDUCER_FILE))


def tabicl_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values from TabICL's own explainer (an all-NaN row as the background).

    The permutation explainer is named explicitly: left to choose, SHAP switches to an
    exact explainer at ten features or fewer, which needs ``2 ** n_features``
    evaluations. Permutation needs ``2 * n_features + 1`` per row. Masked rows go to the
    model in large batches because each predict call pays a fixed cost.

    Args:
        model: a TabICLRegressor fit with ``cache=True`` on raw numeric features; the
            explainer predicts once per explained row.
        X: numeric rows to explain.

    Returns:
        tuple: (shap_values of shape (n_rows, n_features), base_values of shape (n_rows,)).
    """
    from tabicl.shap import get_shap_explainer

    X_np = np.asarray(X, dtype=np.float64)

    def predict(masked: np.ndarray) -> np.ndarray:
        # The model was fit on a named frame, so masked rows go back in as one
        return model.predict(pd.DataFrame(masked, columns=X.columns))

    explainer = get_shap_explainer(model, X_np, predict_fn=predict, algorithm="permutation")
    sv = explainer(X_np, max_evals=2 * X_np.shape[1] + 1, batch_size=1024)
    return np.asarray(sv.values), np.asarray(sv.base_values).reshape(-1)


def probe_inference_memory(model_dir: str, X_sample: pd.DataFrame) -> float:
    """Peak memory (GB) of a fresh CPU-only process that loads the saved model and predicts.

    Serving runs one process per endpoint, so this is what an endpoint must hold. The
    peak is at load, well above what the model settles to, and a container too small
    for it is killed there. Measured rather than estimated: it tracks cache mode,
    ensemble size, feature reduction, and TabICL's own version.

    Args:
        model_dir: directory the model was saved to.
        X_sample: feature rows to predict (the first PROBE_ROWS are used).

    Returns:
        float: peak resident memory in GB.
    """
    code = (
        "import resource, sys\n"
        "import pandas as pd\n"
        "from workbench.endpoints.tabicl_utils import load_tabicl, predict_with_std\n"
        "model, reducer = load_tabicl(sys.argv[1], device='cpu')\n"
        "predict_with_std(model, pd.read_pickle(sys.argv[2]), reducer)\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = os.path.join(tmp, "probe_rows.pkl")
        X_sample.head(PROBE_ROWS).to_pickle(sample_path)
        result = subprocess.run(
            [sys.executable, "-c", code, model_dir, sample_path],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"The saved model failed to load and predict on CPU:\n{result.stderr[-3000:]}")
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    peak = int(result.stdout.strip().splitlines()[-1])
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
