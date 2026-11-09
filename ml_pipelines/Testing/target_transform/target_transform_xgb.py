"""Does a target transform earn its place on MPPB? Three XGBoost models, one answer.

MPPB is percent unbound: physically bounded [0, 100], heavily skewed toward zero, and the
range that matters clinically is under 1%. The three variants separate the two knobs:

    plain    no transform, no bounds -- the baseline
    bounds   prediction_bounds only -- a linear target held inside its physical range
    log10    log10(offset=0.1) + bounds -- the transform the skew argues for

The offset is the assay's smallest non-zero reading, so the 10 zero rows land at -1 in
transformed space rather than failing the domain check.

Metrics come from the pooled out-of-fold predictions, always on the original percent scale,
with the under-1% rows broken out separately.

Models:
    - mppb-xgb-plain, mppb-xgb-bounds, mppb-xgb-log10
Endpoints:
    - mppb-xgb-plain, mppb-xgb-bounds, mppb-xgb-log10
"""

import numpy as np
import pandas as pd

from workbench.api import Endpoint, FeatureSet, Model, ModelType, ModelFramework

FS_NAME = "mppb_raw"
TARGET = "mppb"
BOUNDS = [0.0, 100.0]
LOW_RANGE = 1.0  # the clinically interesting tail, in percent unbound

VARIANTS = {
    "plain": {},
    "bounds": {"prediction_bounds": BOUNDS},
    "log10": {"target_transform": {"name": "log10", "offset": 0.1}, "prediction_bounds": BOUNDS},
}


def metrics(df: pd.DataFrame) -> dict:
    """MAE, RMSE and Spearman on the original percent scale."""
    error = df["prediction"] - df[TARGET]
    return {
        "n": len(df),
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt((error**2).mean()),
        "spearman": df["prediction"].corr(df[TARGET], method="spearman"),
    }


def train(variant: str, hyperparameters: dict, features: list) -> Model:
    name = f"mppb-xgb-{variant}"
    print(f"\n{'=' * 60}\nTraining {name}: {hyperparameters or 'baseline'}\n{'=' * 60}")

    fs = FeatureSet(FS_NAME)
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column=TARGET,
        feature_list=features,
        description=f"MPPB percent unbound, XGBoost ({variant})",
        tags=["target_transform", "mppb", variant],
        hyperparameters={"n_folds": 5, **hyperparameters},
    )
    model.set_owner("test")

    end = model.to_endpoint(tags=["target_transform", "mppb", variant])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
    return model


def main():
    features = Endpoint("smiles-to-2d-v1").output_columns()
    results = {}

    for variant, hyperparameters in VARIANTS.items():
        model = train(variant, hyperparameters, features)
        oof = model.get_inference_predictions("full_cross_fold")
        results[variant] = {
            "all": metrics(oof),
            "low": metrics(oof[oof[TARGET] < LOW_RANGE]),
            "range": (oof["prediction"].min(), oof["prediction"].max()),
        }

    print("\n" + "=" * 78)
    print("MPPB TARGET TRANSFORM COMPARISON (out-of-fold, percent unbound)")
    print("=" * 78)
    header = f"{'variant':<10}{'MAE':>9}{'RMSE':>9}{'spearman':>11}{'MAE <1%':>10}{'pred range':>22}"
    print(header)
    print("-" * 78)
    for variant, r in results.items():
        pred_range = f"[{r['range'][0]:.2f}, {r['range'][1]:.2f}]"
        print(
            f"{variant:<10}{r['all']['mae']:>9.3f}{r['all']['rmse']:>9.3f}"
            f"{r['all']['spearman']:>11.3f}{r['low']['mae']:>10.3f}{pred_range:>22}"
        )
    print("-" * 78)
    low_n = results["plain"]["low"]["n"]
    print(f"MAE <1% is over the {low_n} rows measured under {LOW_RANGE:g}% unbound")
    print("A prediction range topping out near 2 would mean an inverse never ran")


if __name__ == "__main__":
    main()
