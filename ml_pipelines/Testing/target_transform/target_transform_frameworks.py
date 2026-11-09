"""Every framework that claims target_transform support has to invert it.

One `log10(offset=0.1)` + bounds model per framework on the as-measured MPPB FeatureSet.
XGBoost is covered by target_transform_xgb.py; this covers the other four, including the
multi-target chemprop path where each target resolves its own transform.

A missing inverse is unmistakable: MPPB runs 0 to 87.6 percent unbound, so predictions left
in transformed space top out near log10(87.7) = 1.94. Each model's prediction range is
printed and checked against the target's own.

Models/Endpoints:
    - mppb-pytorch-log10, mppb-chemprop-log10, mppb-tabicl-log10, mppb-atomjepa-log10
    - mppb-chemprop-mt-log10 (multi-target: mppb + a per-target transform)
"""

import argparse

from workbench.api import Endpoint, FeatureSet, Model, ModelType, ModelFramework

FS_NAME = "mppb_raw"
TARGET = "mppb"
BOUNDS = [0.0, 100.0]
TRANSFORM = {"name": "log10", "offset": 0.1}
TRANSFORM_HYPERPARAMETERS = {"target_transform": TRANSFORM, "prediction_bounds": BOUNDS}

# A transformed prediction would land in log10 space, roughly [-1, 2]
MIN_CREDIBLE_MAX = 5.0


def check_scale(model: Model) -> None:
    """Report whether out-of-fold predictions came back on the target's own scale."""
    oof = model.get_inference_predictions("full_cross_fold")
    if oof is None:
        print(f"  {model.name}: no out-of-fold predictions to check")
        return

    low, high = oof["prediction"].min(), oof["prediction"].max()
    verdict = "ORIGINAL SCALE" if high >= MIN_CREDIBLE_MAX else "*** LOOKS TRANSFORMED ***"
    print(f"  {model.name}: predictions [{low:.2f}, {high:.2f}] vs target [0, 87.6] -> {verdict}")


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

    trained.append(build("mppb-pytorch-log10", ModelFramework.PYTORCH, tabular, TARGET, TRANSFORM_HYPERPARAMETERS))
    trained.append(build("mppb-chemprop-log10", ModelFramework.CHEMPROP, ["smiles"], TARGET, TRANSFORM_HYPERPARAMETERS))
    trained.append(build("mppb-tabicl-log10", ModelFramework.TABICL, tabular, TARGET, TRANSFORM_HYPERPARAMETERS))

    # Multi-target, mapping form: mppb is transformed and bounded, mollogp is left alone.
    # One target transformed and one not is the case a single shared spec cannot express.
    trained.append(
        build(
            "mppb-chemprop-mt-log10",
            ModelFramework.CHEMPROP,
            ["smiles"],
            [TARGET, "mollogp"],
            {"target_transform": {TARGET: TRANSFORM}, "prediction_bounds": {TARGET: BOUNDS}},
        )
    )

    if not args.skip_atomjepa:
        trained.append(
            build(
                "mppb-atomjepa-log10",
                ModelFramework.ATOMJEPA,
                ["smiles"],
                TARGET,
                TRANSFORM_HYPERPARAMETERS,
                async_endpoint=True,
            )
        )

    print("\n" + "=" * 78)
    print("PREDICTION SCALE CHECK")
    print("=" * 78)
    for model in trained:
        check_scale(model)


if __name__ == "__main__":
    main()
