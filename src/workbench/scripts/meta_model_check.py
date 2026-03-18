"""MetaModelCheck: Compare simulated ensemble performance vs actual meta endpoint results.

CLI tool that takes a meta model endpoint, looks up its child endpoints,
retrieves the deployed configuration (strategy, weights, corr_scale), and
reproduces the template's aggregation logic on captured child predictions.
Compares the reproduced predictions against the actual meta endpoint captures.
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats

from workbench.api import Endpoint, Model


def compare_results(sim_df: pd.DataFrame, actual_df: pd.DataFrame, id_column: str, target_column: str):
    """Compare simulated ensemble predictions vs actual meta endpoint predictions.

    Args:
        sim_df (pd.DataFrame): Simulated ensemble predictions
        actual_df (pd.DataFrame): Actual meta endpoint inference predictions
        id_column (str): Column to join on
        target_column (str): Target column name
    """
    # Merge on id column
    merged = sim_df.merge(actual_df, on=id_column, suffixes=("_sim", "_actual"))
    print(f"\nMatched {len(merged)} rows (sim: {len(sim_df)}, actual: {len(actual_df)})")

    if len(merged) == 0:
        print("ERROR: No matching rows found. Check that both use the same id_column and data.")
        return

    # Target column might be duplicated with suffixes or just appear once
    if f"{target_column}_sim" in merged.columns:
        target = merged[f"{target_column}_sim"]
    elif target_column in merged.columns:
        target = merged[target_column]
    else:
        print(f"WARNING: Target column '{target_column}' not found in merged data")
        target = None

    pred_sim = merged["prediction_sim"]
    pred_actual = merged["prediction_actual"]

    # Header
    print("\n" + "=" * 70)
    print("SIMULATED vs ACTUAL META ENDPOINT COMPARISON")
    print("=" * 70)

    # Prediction comparison
    print("\n--- Prediction Comparison ---")
    diff = pred_sim - pred_actual
    print(f"  Mean diff (sim - actual):   {diff.mean():.6f}")
    print(f"  Std diff:                   {diff.std():.6f}")
    print(f"  Max |diff|:                 {diff.abs().max():.6f}")
    print(f"  Median |diff|:              {diff.abs().median():.6f}")

    corr, _ = stats.pearsonr(pred_sim, pred_actual)
    print(f"  Pearson r (sim vs actual):  {corr:.6f}")

    # Performance vs target
    if target is not None:
        print("\n--- Performance vs Target ---")
        sim_mae = (pred_sim - target).abs().mean()
        actual_mae = (pred_actual - target).abs().mean()
        sim_rmse = np.sqrt(((pred_sim - target) ** 2).mean())
        actual_rmse = np.sqrt(((pred_actual - target) ** 2).mean())

        print(f"  {'Metric':<12} {'Simulated':>12} {'Actual':>12} {'Diff':>12}")
        print(f"  {'-'*48}")
        print(f"  {'MAE':<12} {sim_mae:>12.4f} {actual_mae:>12.4f} {sim_mae - actual_mae:>+12.4f}")
        print(f"  {'RMSE':<12} {sim_rmse:>12.4f} {actual_rmse:>12.4f} {sim_rmse - actual_rmse:>+12.4f}")

    # Confidence comparison
    if "confidence_sim" in merged.columns and "confidence_actual" in merged.columns:
        conf_sim = merged["confidence_sim"]
        conf_actual = merged["confidence_actual"]

        print("\n--- Confidence Comparison ---")
        print(f"  Sim confidence:    mean={conf_sim.mean():.4f}, std={conf_sim.std():.4f}")
        print(f"  Actual confidence: mean={conf_actual.mean():.4f}, std={conf_actual.std():.4f}")

        conf_diff = conf_sim - conf_actual
        print(f"  Mean |conf diff|:  {conf_diff.abs().mean():.6f}")
        print(f"  Max |conf diff|:   {conf_diff.abs().max():.6f}")

        conf_corr, _ = stats.pearsonr(conf_sim, conf_actual)
        print(f"  Pearson r (sim vs actual confidence): {conf_corr:.4f}")

        if target is not None:
            sim_abs_err = (pred_sim - target).abs()
            actual_abs_err = (pred_actual - target).abs()

            sim_conf_err_corr = stats.spearmanr(conf_sim, sim_abs_err)[0]
            actual_conf_err_corr = stats.spearmanr(conf_actual, actual_abs_err)[0]
            print(f"\n  Conf-to-error correlation (Spearman):")
            print(f"    Simulated: {sim_conf_err_corr:.4f}")
            print(f"    Actual:    {actual_conf_err_corr:.4f}")

    # Zero prediction check
    n_zero_sim = (pred_sim.abs() < 1e-10).sum()
    n_zero_actual = (pred_actual.abs() < 1e-10).sum()
    if n_zero_sim > 0 or n_zero_actual > 0:
        print(f"\n--- Zero Prediction Check ---")
        print(f"  Simulated zeros:  {n_zero_sim}")
        print(f"  Actual zeros:     {n_zero_actual}")

    # Rows where sim and actual disagree most
    merged["abs_diff"] = diff.abs()
    worst = merged.nlargest(5, "abs_diff")
    print(f"\n--- Top 5 Largest Disagreements ---")
    cols = [id_column, "prediction_sim", "prediction_actual", "abs_diff"]
    if target is not None and target_column in merged.columns:
        cols.insert(1, target_column)
    elif f"{target_column}_sim" in merged.columns:
        cols.insert(1, f"{target_column}_sim")
    print(worst[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulated ensemble performance vs actual meta endpoint results."
    )
    parser.add_argument(
        "meta_endpoint",
        help="Name of the meta model endpoint to check.",
    )
    parser.add_argument(
        "--capture-name",
        default="full_cross_fold",
        help="Inference capture name (default: full_cross_fold)",
    )
    args = parser.parse_args()

    # Look up the meta endpoint → model → child endpoints
    print(f"Looking up meta endpoint: {args.meta_endpoint}")
    endpoint = Endpoint(args.meta_endpoint)
    model_name = endpoint.get_input()

    from workbench.api.meta_model import MetaModel

    meta_model = MetaModel(model_name)
    meta = meta_model.workbench_meta()

    endpoints = meta.get("endpoints")
    if not endpoints:
        print(f"ERROR: No 'endpoints' found in workbench_meta for model '{model_name}'")
        return

    target_column = meta.get("workbench_model_target")
    print(f"Meta model: {model_name}")
    print(f"Child endpoints: {endpoints}")
    print(f"Target column: {target_column}")

    # Retrieve the deployed meta config (strategy, weights, corr_scale)
    print(f"\n{'='*70}")
    print("RETRIEVING DEPLOYED META CONFIG")
    print(f"{'='*70}")
    meta_config = meta_model.get_meta_config()
    print(f"  Aggregation strategy: {meta_config['aggregation_strategy']}")
    print(f"  Model weights: {meta_config['model_weights']}")
    print(f"  Corr scale: {meta_config.get('corr_scale', {})}")

    # Build endpoint → model name mapping
    ep_to_model = {ep: Endpoint(ep).get_input() for ep in endpoints}
    print(f"  Endpoint → Model mapping: {ep_to_model}")

    # Run simulation on child models
    print(f"\n{'='*70}")
    print("REPRODUCING DEPLOYED AGGREGATION FROM CHILD CAPTURES")
    print(f"{'='*70}")
    from workbench.utils.meta_model_simulator import MetaModelSimulator

    id_column = MetaModel._resolve_id_column(endpoints[0])
    model_names = list(ep_to_model.values())
    sim = MetaModelSimulator(model_names, id_column=id_column, capture_name=args.capture_name)

    # Reproduce the deployed template's exact logic
    sim_df = sim.reproduce_deployed(
        aggregation_strategy=meta_config["aggregation_strategy"],
        model_weights=meta_config["model_weights"],
        corr_scale=meta_config.get("corr_scale"),
        endpoint_to_model=ep_to_model,
    )
    print(f"Simulated predictions: {len(sim_df)} rows")

    # Get actual meta endpoint inference predictions
    print(f"\n{'='*70}")
    print("LOADING ACTUAL META ENDPOINT PREDICTIONS")
    print(f"{'='*70}")
    actual_df = meta_model.get_inference_predictions(args.capture_name)
    if actual_df is None:
        print(f"ERROR: No '{args.capture_name}' predictions found for meta model '{model_name}'")
        print("Run inference on the meta endpoint first.")
        return

    print(f"Actual predictions: {len(actual_df)} rows")

    # Compare
    compare_results(sim_df, actual_df, id_column, target_column)


if __name__ == "__main__":
    main()
