"""Compare different dataset splitting strategies for ChemProp regression models.

This script trains ChemProp models using random, scaffold, and butina splitting
strategies, then compares the cross-fold validation results to see how much
splitting strategy affects model metrics.
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Configuration
feature_set_name = "open_admet_all"
target = "logd"

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# Split strategies to compare
split_strategies = ["random", "scaffold", "butina"]

# Train a model for each split strategy
for strategy in split_strategies:
    model_name = f"logd-chemprop-split-{strategy}"

    # Create the model with the specified split strategy
    if recreate or not Model(model_name).exists():
        print(f"\n{'=' * 60}")
        print(f"Training model with {strategy} split: {model_name}")
        print(f"{'=' * 60}")
        feature_set = FeatureSet(feature_set_name)
        model = feature_set.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=target,
            feature_list=["smiles"],
            description=f"ChemProp Regression with {strategy} split",
            tags=["split-comparison", "chemprop", strategy],
            hyperparameters={
                "split_strategy": strategy,
            },
        )

    # Create an Endpoint for the Regression Model
    if recreate or not Endpoint(model_name).exists():
        m = Model(model_name)
        end = m.to_endpoint(tags=["split", "chemprop", strategy])
        end.set_owner("BW")

        # Run inference on the endpoint
        end.auto_inference()
        end.cross_fold_inference()

# Compare results
print("\n" + "=" * 80)
print("SPLIT STRATEGY COMPARISON (ChemProp) - Cross-Fold Validation Results")
print("=" * 80)

results = {}
for strategy in split_strategies:
    model_name = f"logd-chemprop-split-{strategy}"
    model = Model(model_name)

    # Get the cross-fold metrics
    metrics = model.get_inference_metrics("full_cross_fold").reset_index().to_dict(orient="records")[0]
    results[strategy] = metrics

    print(f"\n{strategy.upper()} Split:")
    print(
        f"  R²:   {metrics.get('r2', 'N/A'):.4f}"
        if isinstance(metrics.get("r2"), (int, float))
        else f"  R²:   {metrics.get('r2', 'N/A')}"
    )
    print(
        f"  MAE:  {metrics.get('mae', 'N/A'):.4f}"
        if isinstance(metrics.get("mae"), (int, float))
        else f"  MAE:  {metrics.get('mae', 'N/A')}"
    )
    print(
        f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}"
        if isinstance(metrics.get("rmse"), (int, float))
        else f"  RMSE: {metrics.get('rmse', 'N/A')}"
    )

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nExpected: Random splits typically show inflated metrics compared to")
print("scaffold/butina splits, which better simulate real-world prediction scenarios")
print("where test molecules are structurally different from training molecules.")
