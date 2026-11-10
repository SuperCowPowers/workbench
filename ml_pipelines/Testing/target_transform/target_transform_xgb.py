"""Does a target transform earn its place on MPPB? Five XGBoost models, one answer.

MPPB is percent unbound: physically bounded [0, 100], heavily skewed toward zero, and the
range that matters clinically is under 1%. The variants separate the two knobs and then cover
every transform on one target:

    plain    no transform, no bounds -- the baseline
    bounds   prediction_bounds only -- a linear target held inside its physical range
    log10    log10(offset=0.1) + bounds -- resolution on the low tail
    logit    logit(ceiling=100) on mppb_floored -- both tails; its inverse needs no bounds
    arcsinh  arcsinh(scale=0.1) + bounds -- same low-tail stretch, no domain to respect

The assay's smallest non-zero reading, 0.1, is what the 10 zero rows need. log10 takes it as
an offset and arcsinh as the scale where its linear region ends, both inside the model. A
logit has no such knob, so it trains on `mppb_floored`: the same target with those 10 rows
clamped in the FeatureSet. That is the one row of this table scored against a truth column
that differs at all, and only on those sub-LLOQ rows.

Metrics come from the pooled out-of-fold predictions, always on the original percent scale,
with the under-1% rows broken out separately.

Models:
    - mppb-xgb-plain, mppb-xgb-bounds, mppb-xgb-log10, mppb-xgb-logit, mppb-xgb-arcsinh
Endpoints:
    - mppb-xgb-plain, mppb-xgb-bounds, mppb-xgb-log10, mppb-xgb-logit, mppb-xgb-arcsinh
"""

import numpy as np
import pandas as pd

from workbench.api import Endpoint, FeatureSet, Model, ModelType, ModelFramework
from workbench.utils.target_transform import resolve_target_transform

FS_NAME = "mppb_raw"
TARGET = "mppb"
BOUNDS = [0.0, 100.0]
LOW_RANGE = 1.0  # the clinically interesting tail, in percent unbound
LLOQ = 0.1  # the assay's smallest non-zero reading
FLOORED_TARGET = "mppb_floored"  # mppb with its 10 zeros clamped to the LLOQ

VARIANTS = {
    "plain": {},
    "bounds": {"prediction_bounds": BOUNDS},
    "log10": {"target_transform": {"name": "log10", "offset": LLOQ}, "prediction_bounds": BOUNDS},
    # A logit inverts into (0, ceiling), so bounds would only add a mass point
    "logit": {"target_transform": {"name": "logit", "ceiling": 100.0}},
    "arcsinh": {"target_transform": {"name": "arcsinh", "scale": LLOQ}, "prediction_bounds": BOUNDS},
}

# The target each variant trains on, where it is not the as-measured column
VARIANT_TARGETS = {"logit": FLOORED_TARGET}


def transformed_max(spec, target_max: float) -> float:
    """Where the target's top lands in this variant's transformed space.

    A prediction range confined inside it means an inverse never ran. With no transform it is
    the target's own top.
    """
    return float(resolve_target_transform(spec).forward(np.array([target_max]))[0])


def metrics(df: pd.DataFrame, target: str) -> dict:
    """MAE, RMSE and Spearman on the original percent scale."""
    error = df["prediction"] - df[target]
    return {
        "n": len(df),
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt((error**2).mean()),
        "spearman": df["prediction"].corr(df[target], method="spearman"),
    }


def train(variant: str, hyperparameters: dict, features: list, target: str) -> Model:
    name = f"mppb-xgb-{variant}"
    print(f"\n{'=' * 60}\nTraining {name}: {hyperparameters or 'baseline'}\n{'=' * 60}")

    fs = FeatureSet(FS_NAME)
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column=target,
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
        target = VARIANT_TARGETS.get(variant, TARGET)
        model = train(variant, hyperparameters, features, target)
        oof = model.get_inference_predictions("full_cross_fold")
        results[variant] = {
            "all": metrics(oof, target),
            "low": metrics(oof[oof[target] < LOW_RANGE], target),
            "range": (oof["prediction"].min(), oof["prediction"].max()),
            "target_max": oof[target].max(),
            "transformed_max": transformed_max(hyperparameters.get("target_transform"), oof[target].max()),
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

    # Each transform compresses the target into a different range, so what an un-inverted
    # prediction would look like is per-variant
    tops = ", ".join(
        f"{variant} {r['transformed_max']:.2f}"
        for variant, r in results.items()
        if VARIANTS[variant].get("target_transform")
    )
    print(f"Target max is {results['plain']['target_max']:g}; transformed space tops out at {tops},")
    print("so a prediction range inside those would mean an inverse never ran")


if __name__ == "__main__":
    main()
