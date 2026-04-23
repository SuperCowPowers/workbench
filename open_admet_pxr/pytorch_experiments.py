"""PyTorch hyperparameter grid search on the PXR pec50 2d_3d feature set.

Separate from all_models.py so the main production sweep stays clean.

Define a grid in PARAM_GRID (dict of hyperparameter -> list of values). The
script expands the cartesian product, trains one PyTorch UQ ensemble per cell,
deploys an endpoint, and kicks off cross-fold inference so each model gets a
validation_predictions.csv with standard metrics + confidence.

Each cell is named ``pxr-2d-3d-reg-pytorch-<N>-<suffix>`` where the suffix
encodes the grid coordinates (short aliases; see _suffix_for). Models are
idempotent (skipped if they exist; --rebuild forces).

After training finishes, `summarize()` pulls validation_predictions.csv for
every grid cell and prints a leaderboard sorted by your chosen metric.

Usage:
    python pytorch_experiments.py                # train any missing grid cells
    python pytorch_experiments.py --rebuild      # force-rebuild every cell
    python pytorch_experiments.py --summarize    # skip training, just tabulate results
"""

import argparse
import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from workbench.api import FeatureEndpoint, FeatureSet, Model, ModelFramework, ModelType

log = logging.getLogger("openadmet_pxr.pytorch_experiments")

# ─── Feature set / target ───────────────────────────────────────────────────
FEATURE_SET = "openadmet_pxr_activity_2d_3d"
TARGET_COL = "pec50"
TAGS_BASE = ["openadmet_pxr", "activity", "regression", "pytorch_experiment"]

ENDPOINT_2D = "smiles-to-2d-v1"
ENDPOINT_3D = "smiles-to-3d-full-v1"


# ─── Grid definition ────────────────────────────────────────────────────────
# Keep the grid small — cell count multiplies fast. Every value in every list
# must be JSON-serializable (passed as hyperparameters to the template).

PARAM_GRID: dict[str, list[Any]] = {
    "dropout": [0.05, 0.15],
    "weight_decay": [1e-4, 1e-3],
    "split_strategy": ["scaffold", "butina"],
}

# Short aliases for name-generation. Any hparam not listed falls back to repr().
SUFFIX_ALIASES: dict[str, dict[Any, str]] = {
    "dropout": {0.0: "d0", 0.05: "d05", 0.1: "d10", 0.15: "d15", 0.2: "d20"},
    "weight_decay": {0.0: "wd0", 1e-5: "wd5", 1e-4: "wd4", 1e-3: "wd3"},
    "split_strategy": {"scaffold": "scaf", "butina": "but", "random": "rand"},
    "restore_best_weights": {True: "rbT", False: "rbF"},
    "loss": {"L1Loss": "l1", "MSELoss": "l2", "HuberLoss": "hub", "SmoothL1Loss": "sl1"},
}


def _suffix_for(cell: dict[str, Any]) -> str:
    """Compose a short, readable suffix from a grid cell."""
    parts = []
    for k, v in cell.items():
        alias = SUFFIX_ALIASES.get(k, {}).get(v, str(v).replace(".", "_"))
        parts.append(alias)
    return "-".join(parts)


def _grid_cells() -> list[dict[str, Any]]:
    """Cartesian product of PARAM_GRID → list of hparam dicts."""
    keys = list(PARAM_GRID.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(PARAM_GRID[k] for k in keys))]


# ─── Helpers ────────────────────────────────────────────────────────────────


def _feature_list_2d_3d() -> list[str]:
    """Assemble the 2d_3d FeatureSet's feature column list."""
    features: list[str] = []
    for ep in (ENDPOINT_2D, ENDPOINT_3D):
        features.extend(FeatureEndpoint(ep).feature_list())
    return features


def _model_name(suffix: str, n_features: int) -> str:
    return f"pxr-2d-3d-reg-pytorch-{n_features}-{suffix}"


def _deploy_and_validate(model) -> None:
    """Create the endpoint and run cross-fold inference."""
    end = model.to_endpoint(tags=TAGS_BASE + ["pytorch"], max_concurrency=1)
    end.set_owner("open_admet_pxr")
    end.cross_fold_inference()


def _train_cell(cell: dict[str, Any], feature_cols: list[str], rebuild: bool) -> None:
    suffix = _suffix_for(cell)
    model_name = _model_name(suffix, len(feature_cols))

    if not (rebuild or not Model(model_name).exists()):
        log.info(f"Skipping {model_name} (exists; use --rebuild to force)")
        return

    log.info(f"\n=== {model_name} ===")
    log.info(f"    hparams: {cell}")

    fs = FeatureSet(FEATURE_SET)
    pyt = fs.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.PYTORCH,
        target_column=TARGET_COL,
        feature_list=feature_cols,
        description=f"PXR pEC50 PyTorch UQ — grid cell {cell}",
        tags=TAGS_BASE + [suffix],
        hyperparameters=cell,
    )
    pyt.set_owner("open_admet_pxr")
    _deploy_and_validate(pyt)


def _train_grid(rebuild: bool) -> None:
    fs = FeatureSet(FEATURE_SET)
    if not fs.exists():
        raise RuntimeError(f"FeatureSet '{FEATURE_SET}' not found. Run create_feature_sets.py first.")

    feature_cols = _feature_list_2d_3d()
    cells = _grid_cells()
    log.info(f"Grid has {len(cells)} cell(s) on {FEATURE_SET} ({len(feature_cols)} features)")

    for cell in cells:
        _train_cell(cell, feature_cols, rebuild=rebuild)


# ─── Summarize ──────────────────────────────────────────────────────────────


def _metrics_for_model(model_name: str) -> dict[str, Any] | None:
    """Pull validation_predictions.csv for a trained cell and compute UQ metrics."""
    m = Model(model_name)
    df = m.get_inference_predictions("model_training")
    if df is None:
        return None

    target = next(
        (
            c
            for c in df.columns
            if c not in {"molecule_name", "smiles", "prediction", "prediction_std", "confidence"}
            and not c.startswith("q_")
        ),
        None,
    )
    if target is None or "prediction" not in df.columns or "prediction_std" not in df.columns:
        return None

    y, p, s = df[target].values, df["prediction"].values, df["prediction_std"].values
    residual = np.abs(y - p)
    spearman_sr, _ = spearmanr(s, residual)

    return {
        "model": model_name,
        "mae": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "spearman_std_res": float(spearman_sr),
        "std_ratio": float(p.std() / y.std()) if y.std() > 0 else float("nan"),
    }


def summarize(sort_by: str = "spearman_std_res") -> pd.DataFrame:
    """Print a leaderboard across all grid cells, sorted by `sort_by` (descending)."""
    feature_cols = _feature_list_2d_3d()
    rows = []
    for cell in _grid_cells():
        suffix = _suffix_for(cell)
        model_name = _model_name(suffix, len(feature_cols))
        metrics = _metrics_for_model(model_name)
        if metrics is None:
            log.info(f"  [skip] {model_name}: no validation predictions yet")
            continue
        rows.append({**cell, **metrics})

    if not rows:
        log.info("No grid cells have validation_predictions.csv yet.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n" + df.to_string(index=False))
    return df


# ─── Entrypoint ─────────────────────────────────────────────────────────────


def main(rebuild: bool, summarize_only: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    if summarize_only:
        summarize()
        return

    _train_grid(rebuild=rebuild)
    log.info("\nAll cells submitted. Summarize results with:")
    log.info("    python pytorch_experiments.py --summarize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild every grid cell.")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Skip training; print a leaderboard of existing grid cells' UQ metrics.",
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild, summarize_only=args.summarize)
