"""Analysis script for MetaModel child model predictions.

This script analyzes the cross-validation predictions from child models to understand:
1. Confidence distributions and their correlation with residuals
2. Correlation of residuals between models
3. Other diagnostic metrics for ensemble design
"""

import pandas as pd
import numpy as np
from scipy import stats
from workbench.api import Model

# Child models to analyze
CHILD_MODELS = ["logd-reg-xgb", "logd-reg-pytorch", "logd-reg-chemprop"]

id = "molecule_name"


def load_cv_predictions(model_name: str) -> pd.DataFrame:
    """Load cross-validation predictions for a model."""
    model = Model(model_name)
    df = model.get_inference_predictions("full_cross_fold")
    df["residual"] = df["prediction"] - df[model.target()]
    df["abs_residual"] = df["residual"].abs()
    return df


def analyze_confidence_vs_residuals(dfs: dict[str, pd.DataFrame]):
    """Analyze how confidence correlates with prediction accuracy."""
    print("=" * 70)
    print("1. CONFIDENCE VS RESIDUALS ANALYSIS")
    print("=" * 70)

    for name, df in dfs.items():
        print(f"\n{name}:")
        print("-" * 50)

        # Basic confidence stats
        conf = df["confidence"]
        print(
            f"  Confidence: mean={conf.mean():.3f}, std={conf.std():.3f}, "
            f"min={conf.min():.3f}, max={conf.max():.3f}"
        )

        # Correlation between confidence and absolute residual
        # We expect NEGATIVE correlation (high confidence = low error)
        corr_pearson, p_pearson = stats.pearsonr(df["confidence"], df["abs_residual"])
        corr_spearman, p_spearman = stats.spearmanr(df["confidence"], df["abs_residual"])

        print("  Confidence vs |residual|:")
        print(f"    Pearson r={corr_pearson:.3f} (p={p_pearson:.2e})")
        print(f"    Spearman r={corr_spearman:.3f} (p={p_spearman:.2e})")

        # Bin by confidence quartiles and show error stats
        df["conf_quartile"] = pd.qcut(df["confidence"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        quartile_stats = df.groupby("conf_quartile", observed=True)["abs_residual"].agg(
            ["mean", "median", "std", "count"]
        )
        print("  Error by confidence quartile:")
        print(quartile_stats.to_string().replace("\n", "\n    "))


def analyze_residual_correlations(dfs: dict[str, pd.DataFrame]):
    """Analyze correlation of residuals between models."""
    print("\n" + "=" * 70)
    print("2. RESIDUAL CORRELATIONS BETWEEN MODELS")
    print("=" * 70)

    # Build a combined DataFrame of residuals
    residual_df = pd.DataFrame()
    for name, df in dfs.items():
        if residual_df.empty:
            residual_df[id] = df[id]
        residual_df[name] = df["residual"].values

    # Correlation matrix
    corr_matrix = residual_df[list(dfs.keys())].corr()
    print("\nPearson correlation of residuals:")
    print(corr_matrix.to_string())

    # Spearman correlation
    spearman_matrix = residual_df[list(dfs.keys())].corr(method="spearman")
    print("\nSpearman correlation of residuals:")
    print(spearman_matrix.to_string())

    print("\nInterpretation:")
    print("  - Low correlation = models make different errors (good for ensemble)")
    print("  - High correlation = models make similar errors (less ensemble benefit)")


def analyze_model_agreement(dfs: dict[str, pd.DataFrame]):
    """Analyze where models agree/disagree in predictions."""
    print("\n" + "=" * 70)
    print("3. MODEL AGREEMENT ANALYSIS")
    print("=" * 70)

    # Build prediction DataFrame
    pred_df = pd.DataFrame()
    for name, df in dfs.items():
        if pred_df.empty:
            pred_df[id] = df[id]
            pred_df["target"] = df[Model(name).target()]
        pred_df[f"{name}_pred"] = df["prediction"].values
        pred_df[f"{name}_conf"] = df["confidence"].values

    # Prediction std across models (disagreement measure)
    pred_cols = [f"{name}_pred" for name in dfs.keys()]
    pred_df["pred_std"] = pred_df[pred_cols].std(axis=1)
    pred_df["pred_mean"] = pred_df[pred_cols].mean(axis=1)
    pred_df["ensemble_residual"] = pred_df["pred_mean"] - pred_df["target"]
    pred_df["ensemble_abs_residual"] = pred_df["ensemble_residual"].abs()

    print("\nPrediction std across models (disagreement):")
    print(
        f"  mean={pred_df['pred_std'].mean():.3f}, median={pred_df['pred_std'].median():.3f}, "
        f"max={pred_df['pred_std'].max():.3f}"
    )

    # Does high disagreement correlate with high error?
    corr, p = stats.spearmanr(pred_df["pred_std"], pred_df["ensemble_abs_residual"])
    print(f"\nDisagreement vs ensemble error: Spearman r={corr:.3f} (p={p:.2e})")

    # Bin by disagreement and show error stats
    pred_df["disagree_quartile"] = pd.qcut(pred_df["pred_std"], q=4, labels=["Q1 (agree)", "Q2", "Q3", "Q4 (disagree)"])
    quartile_stats = pred_df.groupby("disagree_quartile", observed=True)["ensemble_abs_residual"].agg(
        ["mean", "median", "count"]
    )
    print("\nEnsemble error by disagreement quartile:")
    print(quartile_stats.to_string().replace("\n", "\n  "))


def analyze_per_model_performance(dfs: dict[str, pd.DataFrame]):
    """Show per-model performance metrics side by side."""
    print("\n" + "=" * 70)
    print("4. PER-MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    metrics = []
    for name, df in dfs.items():
        residuals = df["residual"]
        target = df[Model(name).target()]
        pred = df["prediction"]

        rmse = np.sqrt((residuals**2).mean())
        mae = residuals.abs().mean()
        r2 = 1 - (residuals**2).sum() / ((target - target.mean()) ** 2).sum()
        spearman = stats.spearmanr(target, pred)[0]

        metrics.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "spearman": spearman,
                "mean_conf": df["confidence"].mean(),
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("model")
    print("\n" + metrics_df.to_string())


def suggest_ensemble_weights(dfs: dict[str, pd.DataFrame]):
    """Suggest model weights based on performance."""
    print("\n" + "=" * 70)
    print("5. SUGGESTED ENSEMBLE WEIGHTS")
    print("=" * 70)

    # Calculate inverse RMSE weights
    rmse_scores = {}
    for name, df in dfs.items():
        residuals = df["residual"]
        rmse_scores[name] = np.sqrt((residuals**2).mean())

    # Inverse RMSE (better models get higher weight)
    inv_rmse = {name: 1.0 / rmse for name, rmse in rmse_scores.items()}
    total = sum(inv_rmse.values())
    weights = {name: w / total for name, w in inv_rmse.items()}

    print("\nWeights based on inverse RMSE:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f} (RMSE={rmse_scores[name]:.3f})")

    # Also show what equal weights would be
    print(f"\nEqual weights would be: {1.0/len(dfs):.3f} each")


def simulate_ensemble_strategies(dfs: dict[str, pd.DataFrame]):
    """Simulate different ensemble strategies and compare performance."""
    print("\n" + "=" * 70)
    print("6. ENSEMBLE STRATEGY COMPARISON")
    print("=" * 70)

    # Build combined DataFrame
    combined = pd.DataFrame()
    model_names = list(dfs.keys())
    first_model = Model(model_names[0])
    target_col = first_model.target()

    for name, df in dfs.items():
        if combined.empty:
            combined[id] = df[id]
            combined["target"] = df[target_col]
        combined[f"{name}_pred"] = df["prediction"].values
        combined[f"{name}_conf"] = df["confidence"].values
        combined[f"{name}_residual"] = df["residual"].values

    pred_cols = [f"{name}_pred" for name in model_names]
    conf_cols = [f"{name}_conf" for name in model_names]

    results = []

    # Strategy 1: Simple mean (equal weights)
    combined["simple_mean"] = combined[pred_cols].mean(axis=1)
    rmse = np.sqrt(((combined["simple_mean"] - combined["target"]) ** 2).mean())
    results.append({"strategy": "Simple Mean", "rmse": rmse})

    # Strategy 2: Confidence-weighted (current meta model approach)
    conf_df = combined[conf_cols].values
    pred_df = combined[pred_cols].values
    conf_sum = conf_df.sum(axis=1, keepdims=True) + 1e-8
    weights = conf_df / conf_sum
    combined["conf_weighted"] = (pred_df * weights).sum(axis=1)
    rmse = np.sqrt(((combined["conf_weighted"] - combined["target"]) ** 2).mean())
    results.append({"strategy": "Confidence-Weighted", "rmse": rmse})

    # Strategy 3: Inverse-RMSE weighted (model-level weights)
    rmse_scores = {name: np.sqrt((dfs[name]["residual"] ** 2).mean()) for name in model_names}
    inv_rmse_weights = np.array([1.0 / rmse_scores[name] for name in model_names])
    inv_rmse_weights = inv_rmse_weights / inv_rmse_weights.sum()
    combined["inv_rmse_weighted"] = (pred_df * inv_rmse_weights).sum(axis=1)
    rmse = np.sqrt(((combined["inv_rmse_weighted"] - combined["target"]) ** 2).mean())
    results.append({"strategy": "Inverse-RMSE Weighted", "rmse": rmse})

    # Strategy 4: Best model only (ChemProp)
    best_model = min(rmse_scores, key=rmse_scores.get)
    combined["best_only"] = combined[f"{best_model}_pred"]
    rmse = np.sqrt(((combined["best_only"] - combined["target"]) ** 2).mean())
    results.append({"strategy": f"Best Model Only ({best_model})", "rmse": rmse})

    # Strategy 5: Confidence-weighted with model-level scaling
    # Scale confidences by inverse RMSE so better models get more weight
    scaled_conf = conf_df * inv_rmse_weights
    scaled_conf_sum = scaled_conf.sum(axis=1, keepdims=True) + 1e-8
    scaled_weights = scaled_conf / scaled_conf_sum
    combined["scaled_conf_weighted"] = (pred_df * scaled_weights).sum(axis=1)
    rmse = np.sqrt(((combined["scaled_conf_weighted"] - combined["target"]) ** 2).mean())
    results.append({"strategy": "Scaled Conf-Weighted (conf * inv_rmse)", "rmse": rmse})

    # Strategy 6: Drop worst model (XGBoost)
    worst_model = max(rmse_scores, key=rmse_scores.get)
    remaining = [n for n in model_names if n != worst_model]
    remaining_pred_cols = [f"{n}_pred" for n in remaining]
    remaining_conf_cols = [f"{n}_conf" for n in remaining]
    rem_conf = combined[remaining_conf_cols].values
    rem_pred = combined[remaining_pred_cols].values
    rem_conf_sum = rem_conf.sum(axis=1, keepdims=True) + 1e-8
    rem_weights = rem_conf / rem_conf_sum
    combined["drop_worst"] = (rem_pred * rem_weights).sum(axis=1)
    rmse = np.sqrt(((combined["drop_worst"] - combined["target"]) ** 2).mean())
    results.append({"strategy": f"Drop Worst ({worst_model})", "rmse": rmse})

    # Print results
    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + results_df.to_string(index=False))

    # Also show individual model RMSEs for reference
    print("\nIndividual model RMSEs for reference:")
    for name, rmse in sorted(rmse_scores.items(), key=lambda x: x[1]):
        print(f"  {name}: {rmse:.4f}")


def analyze_confidence_weight_distribution(dfs: dict[str, pd.DataFrame]):
    """Analyze how confidence weights are distributed across models."""
    print("\n" + "=" * 70)
    print("7. CONFIDENCE WEIGHT DISTRIBUTION")
    print("=" * 70)

    # Build confidence DataFrame
    model_names = list(dfs.keys())
    conf_df = pd.DataFrame()
    for name, df in dfs.items():
        conf_df[name] = df["confidence"].values

    # Calculate weights per row
    conf_sum = conf_df.sum(axis=1)
    weight_df = conf_df.div(conf_sum, axis=0)

    print("\nMean weight per model (from confidence-weighting):")
    for name in model_names:
        print(f"  {name}: {weight_df[name].mean():.3f}")

    print("\nWeight distribution stats:")
    print(weight_df.describe().to_string())

    # How often does each model "win" (have highest weight)?
    print("\nHow often each model has highest weight:")
    winner = weight_df.idxmax(axis=1)
    winner_counts = winner.value_counts()
    for name in model_names:
        count = winner_counts.get(name, 0)
        print(f"  {name}: {count} ({100*count/len(weight_df):.1f}%)")


def analyze_where_ensemble_fails(dfs: dict[str, pd.DataFrame]):
    """Identify cases where ensemble is worse than best individual model."""
    print("\n" + "=" * 70)
    print("8. WHERE ENSEMBLE FAILS")
    print("=" * 70)

    # Build combined DataFrame
    model_names = list(dfs.keys())
    first_model = Model(model_names[0])
    target_col = first_model.target()

    combined = pd.DataFrame()
    for name, df in dfs.items():
        if combined.empty:
            combined[id] = df[id]
            combined["target"] = df[target_col]
        combined[f"{name}_pred"] = df["prediction"].values
        combined[f"{name}_conf"] = df["confidence"].values
        combined[f"{name}_abs_err"] = df["abs_residual"].values

    pred_cols = [f"{name}_pred" for name in model_names]
    conf_cols = [f"{name}_conf" for name in model_names]
    err_cols = [f"{name}_abs_err" for name in model_names]

    # Calculate ensemble prediction (confidence-weighted)
    conf_df = combined[conf_cols].values
    pred_df = combined[pred_cols].values
    conf_sum = conf_df.sum(axis=1, keepdims=True) + 1e-8
    weights = conf_df / conf_sum
    combined["ensemble_pred"] = (pred_df * weights).sum(axis=1)
    combined["ensemble_abs_err"] = (combined["ensemble_pred"] - combined["target"]).abs()

    # Find best individual model error per row
    combined["best_individual_err"] = combined[err_cols].min(axis=1)
    combined["best_individual_model"] = combined[err_cols].idxmin(axis=1).str.replace("_abs_err", "")

    # Cases where ensemble is worse
    combined["ensemble_worse"] = combined["ensemble_abs_err"] > combined["best_individual_err"]
    n_worse = combined["ensemble_worse"].sum()
    n_total = len(combined)

    print(f"\nEnsemble is worse than best individual model in {n_worse}/{n_total} cases ({100*n_worse/n_total:.1f}%)")

    # Analyze these cases
    worse_cases = combined[combined["ensemble_worse"]]
    print("\nIn cases where ensemble fails:")
    print(f"  Mean ensemble error: {worse_cases['ensemble_abs_err'].mean():.3f}")
    print(f"  Mean best individual error: {worse_cases['best_individual_err'].mean():.3f}")
    print(f"  Mean error increase: {(worse_cases['ensemble_abs_err'] - worse_cases['best_individual_err']).mean():.3f}")

    # Which model would have been best in these cases?
    print("\nWhich model would have been best in failed cases:")
    best_counts = worse_cases["best_individual_model"].value_counts()
    for name, count in best_counts.items():
        print(f"  {name}: {count} ({100*count/len(worse_cases):.1f}%)")

    # Is there a pattern with confidence?
    print("\nConfidence patterns in failed cases:")
    for name in model_names:
        mean_conf_all = combined[f"{name}_conf"].mean()
        mean_conf_worse = worse_cases[f"{name}_conf"].mean()
        print(f"  {name}: all={mean_conf_all:.3f}, failed={mean_conf_worse:.3f}")


def merge_dataframes(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Merge all DataFrames on the id column to ensure alignment."""
    # Sort each DataFrame by id and reset index
    return {name: df.sort_values(id).reset_index(drop=True) for name, df in dfs.items()}


if __name__ == "__main__":
    print("Loading cross-validation predictions...")
    dfs = {}
    for name in CHILD_MODELS:
        print(f"  Loading {name}...")
        dfs[name] = load_cv_predictions(name)

    print(f"\nLoaded {len(dfs)} models, {len(list(dfs.values())[0])} samples each\n")

    # Merge on id column to ensure alignment
    dfs = merge_dataframes(dfs)

    # Run analyses
    analyze_confidence_vs_residuals(dfs)
    analyze_residual_correlations(dfs)
    analyze_model_agreement(dfs)
    analyze_per_model_performance(dfs)
    suggest_ensemble_weights(dfs)
    simulate_ensemble_strategies(dfs)
    analyze_confidence_weight_distribution(dfs)
    analyze_where_ensemble_fails(dfs)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
