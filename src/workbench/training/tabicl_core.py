"""TabICL training primitives shared by the tabicl template and local experiments.

TabICL is an in-context model: ``fit`` stores the training rows (optionally caching their
transformer projections) and every prediction conditions on them. Nothing is trained, so
the levers here are what gets cached and how much memory serving needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from workbench.endpoints.tabicl_utils import MODEL_FILE, REDUCER_FILE

# Rows the memory probe predicts: a typical small serving request
PROBE_ROWS = 10


def fit_tabicl(hyperparameters: dict, X: pd.DataFrame, y, *, kv_cache) -> tuple:
    """Fit a TabICL regressor, with the optional feature reducer in front.

    Args:
        hyperparameters: the template's resolved hyperparameters.
        X: training features.
        y: training target.
        kv_cache: False for a throwaway model that predicts once (fold models); the
            ``kv_cache`` hyperparameter for the served model.

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
        kv_cache=kv_cache,
        random_state=hyperparameters["seed"],
        model_path=str(resolve_foundation_checkpoint("tabicl")),
        allow_auto_download=False,
    ).fit(X, y)
    return model, reducer


def save_tabicl(model, reducer, model_dir: str) -> None:
    """Save the served model self-contained: pretrained weights travel in the artifact,
    so an endpoint never reaches for Hugging Face."""
    import joblib

    model.save(os.path.join(model_dir, MODEL_FILE), save_model_weights=True)
    if reducer is not None:
        joblib.dump(reducer, os.path.join(model_dir, REDUCER_FILE))


def tabicl_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values from TabICL's own explainer (an all-NaN row as the background).

    The permutation explainer needs at least ``2 * n_features + 1`` evaluations per
    row. Masked rows go to the model in large batches because each predict call pays a
    fixed cost for the in-context pass.

    Args:
        model: a fitted TabICLRegressor taking raw numeric features.
        X: numeric rows to explain.

    Returns:
        tuple: (shap_values of shape (n_rows, n_features), base_values of shape (n_rows,)).
    """
    from tabicl.shap import get_shap_explainer

    X_np = np.asarray(X, dtype=np.float64)

    def predict(masked: np.ndarray) -> np.ndarray:
        # The model was fit on a named frame, so masked rows go back in as one
        return model.predict(pd.DataFrame(masked, columns=X.columns))

    explainer = get_shap_explainer(model, X_np, predict_fn=predict)
    sv = explainer(X_np, max_evals=2 * X_np.shape[1] + 1, batch_size=1024)
    return np.asarray(sv.values), np.asarray(sv.base_values).reshape(-1)


def probe_inference_memory(model_dir: str, X_sample: pd.DataFrame) -> float:
    """Peak memory (GB) of a fresh CPU-only process that loads the saved model and predicts.

    Serving runs one process per endpoint, so this is what a serverless endpoint must
    hold. Measured rather than estimated: it tracks cache mode, ensemble size, feature
    reduction, and TabICL's own version.

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
            check=True,
        )
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    peak = int(result.stdout.strip().splitlines()[-1])
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
