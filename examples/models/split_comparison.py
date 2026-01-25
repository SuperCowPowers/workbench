"""Compare different dataset splitting strategies for XGBoost regression models.

This script trains XGBoost models using random, scaffold, and butina splitting
strategies, then compares the cross-fold validation results to see how much
splitting strategy affects model metrics.
"""

from workbench.api import FeatureSet, Model, ModelType

# Configuration
feature_set_name = "aqsol_features"
target = "solubility"
feature_list = [
    "molwt",
    "mollogp",
    "molmr",
    "heavyatomcount",
    "numhacceptors",
    "numhdonors",
    "numheteroatoms",
    "numrotatablebonds",
    "numvalenceelectrons",
    "numaromaticrings",
    "numsaturatedrings",
    "numaliphaticrings",
    "ringcount",
    "tpsa",
    "labuteasa",
    "balabanj",
    "bertzct",
]

# Split strategies to compare
split_strategies = ["random", "scaffold", "butina"]

# Train a model for each split strategy
for strategy in split_strategies:
    model_name = f"aqsol-split-{strategy}"
    print(f"\n{'='*60}")
    print(f"Training model with {strategy} split: {model_name}")
    print(f"{'='*60}")

    # Delete existing model if it exists
    model = Model(model_name)
    if model.exists():
        model.delete()

    # Create the model with the specified split strategy
    feature_set = FeatureSet(feature_set_name)
    model = feature_set.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        feature_list=feature_list,
        target_column=target,
        description=f"XGBoost Regression with {strategy} split",
        tags=["split-comparison", strategy],
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 6,
            "split_strategy": strategy,
        },
    )

# Compare results
print("\n" + "=" * 80)
print("SPLIT STRATEGY COMPARISON - Cross-Fold Validation Results")
print("=" * 80)

results = {}
for strategy in split_strategies:
    model_name = f"aqsol-split-{strategy}"
    model = Model(model_name)

    # Get the cross-fold metrics
    metrics = model.get_inference_metrics("full_cross_fold")
    results[strategy] = metrics

    print(f"\n{strategy.upper()} Split:")
    print(f"  R²:   {metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else f"  R²:   {metrics.get('r2', 'N/A')}")
    print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else f"  MAE:  {metrics.get('mae', 'N/A')}")
    print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else f"  RMSE: {metrics.get('rmse', 'N/A')}")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nExpected: Random splits typically show inflated metrics compared to")
print("scaffold/butina splits, which better simulate real-world prediction scenarios")
print("where test molecules are structurally different from training molecules.")
