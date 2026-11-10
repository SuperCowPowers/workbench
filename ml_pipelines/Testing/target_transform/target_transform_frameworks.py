"""Every framework that claims target_transform support has to invert it.

One model per framework on the as-measured MPPB FeatureSet, each carrying a different
transform, so a single run covers both axes. XGBoost is covered by target_transform_xgb.py
alongside its own transform comparison; this covers the other four, including the
multi-target chemprop path where each target resolves its own transform.

    pytorch      logit(ceiling=100, floor=0.1), no bounds -- its inverse is bounded already
    chemprop     arcsinh(scale=0.1) + bounds -- unbounded either way, so bounds earn a place
    tabicl       log10(offset=0.1) + bounds
    atomjepa     log10(offset=0.1) + bounds
    chemprop-mt  logit on mppb, a bare arcsinh on the signed mollogp

A missing inverse is unmistakable, and each transform compresses the target differently:
MPPB runs 0 to 87.6 percent unbound, which lands at 1.94 under log10, 0.85 under logit and
7.47 under arcsinh. Every model's prediction range is printed against the top of its own
transformed space.

Models/Endpoints:
    - mppb-pytorch-logit, mppb-chemprop-arcsinh, mppb-tabicl-log10, mppb-atomjepa-log10
    - mppb-chemprop-mt-logit (multi-target: a different transform on each target)
"""

import argparse

import numpy as np

from workbench.api import Endpoint, FeatureSet, Model, ModelType, ModelFramework
from workbench.utils.target_transform import resolve_target_transform

FS_NAME = "mppb_raw"
TARGET = "mppb"
BOUNDS = [0.0, 100.0]
LLOQ = 0.1  # the assay's smallest non-zero reading

LOG10 = {"name": "log10", "offset": LLOQ}
LOGIT = {"name": "logit", "ceiling": 100.0, "floor": LLOQ}
ARCSINH = {"name": "arcsinh", "scale": LLOQ}

LOG10_HYPERPARAMETERS = {"target_transform": LOG10, "prediction_bounds": BOUNDS}

# Clear of the top of any transformed space, so an un-inverted prediction cannot pass
SCALE_MARGIN = 2.0


def check_scale(model: Model, spec) -> None:
    """Report whether out-of-fold predictions came back on the target's own scale."""
    oof = model.get_inference_predictions("full_cross_fold")
    if oof is None:
        print(f"  {model.name}: no out-of-fold predictions to check")
        return

    target_max = oof[TARGET].max()
    transformed_max = float(resolve_target_transform(spec).forward(np.array([target_max]))[0])
    low, high = oof["prediction"].min(), oof["prediction"].max()
    verdict = "ORIGINAL SCALE" if high > SCALE_MARGIN * transformed_max else "*** LOOKS TRANSFORMED ***"
    print(
        f"  {model.name}: predictions [{low:.2f}, {high:.2f}] vs target [0, {target_max:.1f}] "
        f"(transformed space tops out at {transformed_max:.2f}) -> {verdict}"
    )


def build(name: str, framework, features: list, targets, hyperparameters: dict, async_endpoint: bool = False) -> Model:
    print(f"\n{'=' * 60}\nTraining {name} ({framework.value})\n{'=' * 60}")
    fs = FeatureSet(FS_NAME)
    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=framework,
        target_column=targets,
        feature_list=features,
        description=f"MPPB percent unbound, {framework.value} with a log10 target transform",
        tags=["target_transform", "mppb", framework.value],
        hyperparameters=hyperparameters,
    )
    model.set_owner("test")

    end = model.to_endpoint(tags=["target_transform", "mppb"], async_endpoint=async_endpoint)
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-atomjepa", action="store_true", help="Skip the Atom-JEPA model (GPU pool + async endpoint)"
    )
    args = parser.parse_args()

    tabular = Endpoint("smiles-to-2d-v1").output_columns()
    trained = []

    # A logit inverts into (0, ceiling), so this one carries no bounds
    trained.append(
        (build("mppb-pytorch-logit", ModelFramework.PYTORCH, tabular, TARGET, {"target_transform": LOGIT}), LOGIT)
    )
    trained.append(
        (
            build(
                "mppb-chemprop-arcsinh",
                ModelFramework.CHEMPROP,
                ["smiles"],
                TARGET,
                {"target_transform": ARCSINH, "prediction_bounds": BOUNDS},
            ),
            ARCSINH,
        )
    )
    trained.append((build("mppb-tabicl-log10", ModelFramework.TABICL, tabular, TARGET, LOG10_HYPERPARAMETERS), LOG10))

    # Multi-target, mapping form: a bounded percent takes a logit, and mollogp goes negative,
    # so it takes a bare arcsinh. Different transforms per target is what a shared spec cannot
    # express, and only mppb has a physical range to bound.
    trained.append(
        (
            build(
                "mppb-chemprop-mt-logit",
                ModelFramework.CHEMPROP,
                ["smiles"],
                [TARGET, "mollogp"],
                {"target_transform": {TARGET: LOGIT, "mollogp": "arcsinh"}},
            ),
            LOGIT,
        )
    )

    if not args.skip_atomjepa:
        trained.append(
            (
                build(
                    "mppb-atomjepa-log10",
                    ModelFramework.ATOMJEPA,
                    ["smiles"],
                    TARGET,
                    LOG10_HYPERPARAMETERS,
                    async_endpoint=True,
                ),
                LOG10,
            )
        )

    print("\n" + "=" * 78)
    print("PREDICTION SCALE CHECK")
    print("=" * 78)
    for model, spec in trained:
        check_scale(model, spec)


if __name__ == "__main__":
    main()
