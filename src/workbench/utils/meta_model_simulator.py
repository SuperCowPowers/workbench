"""MetaModelSimulator: Simulate and analyze ensemble model performance.

This class helps evaluate whether a meta model (ensemble) would outperform
individual child models by analyzing endpoint inference predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

from workbench.api import Model

# Set up the log
log = logging.getLogger("workbench")


class MetaModelSimulator:
    """Simulate meta model performance from child model predictions.

    This class loads cross-validation predictions from multiple models and
    analyzes how different ensemble strategies would perform compared to
    the individual models.

    Example:
        ```python
        from workbench.utils.meta_model_simulator import MetaModelSimulator

        sim = MetaModelSimulator(["model-a", "model-b", "model-c"])
        sim.report()  # Print full analysis
        sim.strategy_comparison()  # Compare ensemble strategies
        ```
    """

    def __init__(self, model_names: list[str], id_column: str = "id"):
        """Initialize the simulator with a list of model names.

        Args:
            model_names: List of model names to include in the ensemble
            id_column: Column name to use for row alignment (default: "id")
        """
        self.model_names = model_names
        self.id_column = id_column
        self._dfs: dict[str, pd.DataFrame] = {}
        self._target_column: str | None = None
        self._load_predictions()

    def _load_predictions(self):
        """Load endpoint inference predictions for all models."""
        log.info(f"Loading predictions for {len(self.model_names)} models...")
        for name in self.model_names:
            model = Model(name)
            if self._target_column is None:
                self._target_column = model.target()
            df = model.get_inference_predictions("full_inference")
            if df is None:
                raise ValueError(
                    f"No full_inference predictions found for model '{name}'. Run endpoint inference first."
                )
            df["residual"] = df["prediction"] - df[self._target_column]
            df["abs_residual"] = df["residual"].abs()
            self._dfs[name] = df

        # Find common rows across all models
        id_sets = {name: set(df[self.id_column]) for name, df in self._dfs.items()}
        common_ids = set.intersection(*id_sets.values())
        sizes = ", ".join(f"{name}: {len(ids)}" for name, ids in id_sets.items())
        log.info(f"Row counts before alignment: {sizes} -> common: {len(common_ids)}")
        self._dfs = {name: df[df[self.id_column].isin(common_ids)] for name, df in self._dfs.items()}

        # Align DataFrames by sorting on id column
        self._dfs = {name: df.sort_values(self.id_column).reset_index(drop=True) for name, df in self._dfs.items()}
        log.info(f"Loaded {len(self._dfs)} models, {len(list(self._dfs.values())[0])} samples each")

    def report(self, details: bool = False):
        """Print a comprehensive analysis report

        Args:
            details: Whether to include detailed sections (default: False)
        """
        self.model_performance()
        self.residual_correlations()
        self.strategy_comparison()
        self.ensemble_failure_analysis()
        if details:
            self.confidence_analysis()
            self.model_agreement()
            self.ensemble_weights()
            self.confidence_weight_distribution()

    def confidence_analysis(self) -> dict[str, dict]:
        """Analyze how confidence correlates with prediction accuracy.

        Returns:
            Dict mapping model name to confidence stats
        """
        print("=" * 60)
        print("CONFIDENCE VS RESIDUALS ANALYSIS")
        print("=" * 60)

        results = {}
        for name, df in self._dfs.items():
            print(f"\n{name}:")
            print("-" * 50)

            conf = df["confidence"]
            print(
                f"  Confidence: mean={conf.mean():.3f}, std={conf.std():.3f}, "
                f"min={conf.min():.3f}, max={conf.max():.3f}"
            )

            corr_pearson, p_pearson = stats.pearsonr(df["confidence"], df["abs_residual"])
            corr_spearman, p_spearman = stats.spearmanr(df["confidence"], df["abs_residual"])

            print("  Confidence vs |residual|:")
            print(f"    Pearson r={corr_pearson:.3f} (p={p_pearson:.2e})")
            print(f"    Spearman r={corr_spearman:.3f} (p={p_spearman:.2e})")

            df["conf_quartile"] = pd.qcut(df["confidence"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
            quartile_stats = df.groupby("conf_quartile", observed=True)["abs_residual"].agg(
                ["mean", "median", "std", "count"]
            )
            print("  Error by confidence quartile:")
            print(quartile_stats.to_string().replace("\n", "\n    "))

            results[name] = {
                "mean_conf": conf.mean(),
                "pearson_r": corr_pearson,
                "spearman_r": corr_spearman,
            }

        return results

    def residual_correlations(self) -> pd.DataFrame:
        """Analyze correlation of residuals between models.

        Returns:
            Correlation matrix DataFrame
        """
        print("\n" + "=" * 60)
        print("RESIDUAL CORRELATIONS BETWEEN MODELS")
        print("=" * 60)

        residual_df = pd.DataFrame({name: df["residual"].values for name, df in self._dfs.items()})

        corr_matrix = residual_df.corr()
        print("\nPearson correlation of residuals:")
        print(corr_matrix.to_string())

        spearman_matrix = residual_df.corr(method="spearman")
        print("\nSpearman correlation of residuals:")
        print(spearman_matrix.to_string())

        print("\nInterpretation:")
        print("  - Low correlation = models make different errors (good for ensemble)")
        print("  - High correlation = models make similar errors (less ensemble benefit)")

        return corr_matrix

    def model_agreement(self) -> dict:
        """Analyze where models agree/disagree in predictions.

        Returns:
            Dict with agreement statistics
        """
        print("\n" + "=" * 60)
        print("MODEL AGREEMENT ANALYSIS")
        print("=" * 60)

        pred_df = pd.DataFrame()
        for name, df in self._dfs.items():
            if pred_df.empty:
                pred_df[self.id_column] = df[self.id_column]
                pred_df["target"] = df[self._target_column]
            pred_df[f"{name}_pred"] = df["prediction"].values

        pred_cols = [f"{name}_pred" for name in self._dfs.keys()]
        pred_df["pred_std"] = pred_df[pred_cols].std(axis=1)
        pred_df["pred_mean"] = pred_df[pred_cols].mean(axis=1)
        pred_df["ensemble_residual"] = pred_df["pred_mean"] - pred_df["target"]
        pred_df["ensemble_abs_residual"] = pred_df["ensemble_residual"].abs()

        print("\nPrediction std across models (disagreement):")
        print(
            f"  mean={pred_df['pred_std'].mean():.3f}, median={pred_df['pred_std'].median():.3f}, "
            f"max={pred_df['pred_std'].max():.3f}"
        )

        corr, p = stats.spearmanr(pred_df["pred_std"], pred_df["ensemble_abs_residual"])
        print(f"\nDisagreement vs ensemble error: Spearman r={corr:.3f} (p={p:.2e})")

        pred_df["disagree_quartile"] = pd.qcut(
            pred_df["pred_std"], q=4, labels=["Q1 (agree)", "Q2", "Q3", "Q4 (disagree)"]
        )
        quartile_stats = pred_df.groupby("disagree_quartile", observed=True)["ensemble_abs_residual"].agg(
            ["mean", "median", "count"]
        )
        print("\nEnsemble error by disagreement quartile:")
        print(quartile_stats.to_string().replace("\n", "\n  "))

        return {
            "mean_disagreement": pred_df["pred_std"].mean(),
            "disagreement_error_corr": corr,
        }

    def model_performance(self) -> pd.DataFrame:
        """Show per-model performance metrics.

        Returns:
            DataFrame with performance metrics for each model
        """
        print("\n" + "=" * 60)
        print("PER-MODEL PERFORMANCE SUMMARY")
        print("=" * 60)

        metrics = []
        for name, df in self._dfs.items():
            residuals = df["residual"]
            target = df[self._target_column]
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
        return metrics_df

    def ensemble_weights(self) -> dict[str, float]:
        """Calculate suggested ensemble weights based on inverse MAE.

        Returns:
            Dict mapping model name to suggested weight
        """
        print("\n" + "=" * 60)
        print("SUGGESTED ENSEMBLE WEIGHTS")
        print("=" * 60)

        mae_scores = {name: df["abs_residual"].mean() for name, df in self._dfs.items()}

        inv_mae = {name: 1.0 / mae for name, mae in mae_scores.items()}
        total = sum(inv_mae.values())
        weights = {name: w / total for name, w in inv_mae.items()}

        print("\nWeights based on inverse MAE:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f} (MAE={mae_scores[name]:.3f})")

        print(f"\nEqual weights would be: {1.0/len(self._dfs):.3f} each")

        return weights

    def strategy_comparison(self) -> pd.DataFrame:
        """Compare different ensemble strategies.

        Returns:
            DataFrame with MAE for each strategy, sorted best to worst
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE STRATEGY COMPARISON")
        print("=" * 60)

        combined = pd.DataFrame()
        model_names = list(self._dfs.keys())

        for name, df in self._dfs.items():
            if combined.empty:
                combined[self.id_column] = df[self.id_column]
                combined["target"] = df[self._target_column]
            combined[f"{name}_pred"] = df["prediction"].values
            combined[f"{name}_conf"] = df["confidence"].values

        pred_cols = [f"{name}_pred" for name in model_names]
        conf_cols = [f"{name}_conf" for name in model_names]

        results = []

        # Strategy 1: Simple mean
        combined["simple_mean"] = combined[pred_cols].mean(axis=1)
        mae = (combined["simple_mean"] - combined["target"]).abs().mean()
        results.append({"strategy": "Simple Mean", "mae": mae})

        # Strategy 2: Confidence-weighted
        conf_arr = combined[conf_cols].values
        pred_arr = combined[pred_cols].values
        conf_sum = conf_arr.sum(axis=1, keepdims=True) + 1e-8
        weights = conf_arr / conf_sum
        combined["conf_weighted"] = (pred_arr * weights).sum(axis=1)
        mae = (combined["conf_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Confidence-Weighted", "mae": mae})

        # Strategy 3: Inverse-MAE weighted
        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
        combined["inv_mae_weighted"] = (pred_arr * inv_mae_weights).sum(axis=1)
        mae = (combined["inv_mae_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Inverse-MAE Weighted", "mae": mae})

        # Strategy 4: Best model only
        best_model = min(mae_scores, key=mae_scores.get)
        combined["best_only"] = combined[f"{best_model}_pred"]
        mae = (combined["best_only"] - combined["target"]).abs().mean()
        results.append({"strategy": f"Best Model Only ({best_model})", "mae": mae})

        # Strategy 5: Scaled confidence-weighted (confidence * model_weights)
        scaled_conf = conf_arr * inv_mae_weights
        scaled_conf_sum = scaled_conf.sum(axis=1, keepdims=True) + 1e-8
        scaled_weights = scaled_conf / scaled_conf_sum
        combined["scaled_conf_weighted"] = (pred_arr * scaled_weights).sum(axis=1)
        mae = (combined["scaled_conf_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Scaled Conf-Weighted", "mae": mae})

        # Strategy 6: Drop worst model (use simple mean of remaining, or raw prediction if only 1 left)
        worst_model = max(mae_scores, key=mae_scores.get)
        remaining = [n for n in model_names if n != worst_model]
        remaining_pred_cols = [f"{n}_pred" for n in remaining]
        if len(remaining) == 1:
            # Single model remaining - use raw prediction (same as "Best Model Only")
            combined["drop_worst"] = combined[remaining_pred_cols[0]]
        else:
            # Multiple models remaining - use simple mean
            combined["drop_worst"] = combined[remaining_pred_cols].mean(axis=1)
        mae = (combined["drop_worst"] - combined["target"]).abs().mean()
        results.append({"strategy": f"Drop Worst ({worst_model})", "mae": mae})

        results_df = pd.DataFrame(results).sort_values("mae")
        print("\n" + results_df.to_string(index=False))

        print("\nIndividual model MAEs for reference:")
        for name, mae in sorted(mae_scores.items(), key=lambda x: x[1]):
            print(f"  {name}: {mae:.4f}")

        return results_df

    def confidence_weight_distribution(self) -> pd.DataFrame:
        """Analyze how confidence weights are distributed across models.

        Returns:
            DataFrame with weight distribution statistics
        """
        print("\n" + "=" * 60)
        print("CONFIDENCE WEIGHT DISTRIBUTION")
        print("=" * 60)

        model_names = list(self._dfs.keys())
        conf_df = pd.DataFrame({name: df["confidence"].values for name, df in self._dfs.items()})

        conf_sum = conf_df.sum(axis=1)
        weight_df = conf_df.div(conf_sum, axis=0)

        print("\nMean weight per model (from confidence-weighting):")
        for name in model_names:
            print(f"  {name}: {weight_df[name].mean():.3f}")

        print("\nWeight distribution stats:")
        print(weight_df.describe().to_string())

        print("\nHow often each model has highest weight:")
        winner = weight_df.idxmax(axis=1)
        winner_counts = winner.value_counts()
        for name in model_names:
            count = winner_counts.get(name, 0)
            print(f"  {name}: {count} ({100*count/len(weight_df):.1f}%)")

        return weight_df

    def ensemble_failure_analysis(self) -> dict:
        """Compare best ensemble strategy vs best individual model.

        Returns:
            Dict with comparison statistics
        """
        print("\n" + "=" * 60)
        print("BEST ENSEMBLE VS BEST MODEL COMPARISON")
        print("=" * 60)

        model_names = list(self._dfs.keys())

        combined = pd.DataFrame()
        for name, df in self._dfs.items():
            if combined.empty:
                combined[self.id_column] = df[self.id_column]
                combined["target"] = df[self._target_column]
            combined[f"{name}_pred"] = df["prediction"].values
            combined[f"{name}_conf"] = df["confidence"].values
            combined[f"{name}_abs_err"] = df["abs_residual"].values

        pred_cols = [f"{name}_pred" for name in model_names]
        conf_cols = [f"{name}_conf" for name in model_names]
        pred_arr = combined[pred_cols].values
        conf_arr = combined[conf_cols].values

        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()

        # Compute all ensemble strategies (true ensembles that combine multiple models)
        ensemble_strategies = {}
        ensemble_strategies["Simple Mean"] = combined[pred_cols].mean(axis=1)
        conf_sum = conf_arr.sum(axis=1, keepdims=True) + 1e-8
        ensemble_strategies["Confidence-Weighted"] = (pred_arr * (conf_arr / conf_sum)).sum(axis=1)
        ensemble_strategies["Inverse-MAE Weighted"] = (pred_arr * inv_mae_weights).sum(axis=1)
        scaled_conf = conf_arr * inv_mae_weights
        scaled_conf_sum = scaled_conf.sum(axis=1, keepdims=True) + 1e-8
        ensemble_strategies["Scaled Conf-Weighted"] = (pred_arr * (scaled_conf / scaled_conf_sum)).sum(axis=1)
        worst_model = max(mae_scores, key=mae_scores.get)
        remaining = [n for n in model_names if n != worst_model]
        remaining_cols = [f"{n}_pred" for n in remaining]
        # Only add Drop Worst if it still combines multiple models
        if len(remaining) > 1:
            ensemble_strategies[f"Drop Worst ({worst_model})"] = combined[remaining_cols].mean(axis=1)

        # Find best individual model
        best_model = min(mae_scores, key=mae_scores.get)
        combined["best_model_abs_err"] = combined[f"{best_model}_abs_err"]
        best_model_mae = mae_scores[best_model]

        # Find best true ensemble strategy
        strategy_maes = {name: (preds - combined["target"]).abs().mean() for name, preds in ensemble_strategies.items()}
        best_strategy = min(strategy_maes, key=strategy_maes.get)
        combined["ensemble_pred"] = ensemble_strategies[best_strategy]
        combined["ensemble_abs_err"] = (combined["ensemble_pred"] - combined["target"]).abs()
        ensemble_mae = strategy_maes[best_strategy]

        # Compare
        combined["ensemble_better"] = combined["ensemble_abs_err"] < combined["best_model_abs_err"]
        n_better = combined["ensemble_better"].sum()
        n_total = len(combined)

        print(f"\nBest individual model: {best_model} (MAE={best_model_mae:.4f})")
        print(f"Best ensemble strategy: {best_strategy} (MAE={ensemble_mae:.4f})")
        if ensemble_mae < best_model_mae:
            improvement = (best_model_mae - ensemble_mae) / best_model_mae * 100
            print(f"Ensemble improves over best model by {improvement:.1f}%")
        else:
            degradation = (ensemble_mae - best_model_mae) / best_model_mae * 100
            print(f"No ensemble benefit: best single model outperforms all ensemble strategies by {degradation:.1f}%")

        print("\nPer-row comparison:")
        print(f"  Ensemble wins: {n_better}/{n_total} ({100*n_better/n_total:.1f}%)")
        print(f"  Best model wins: {n_total - n_better}/{n_total} ({100*(n_total - n_better)/n_total:.1f}%)")

        # When ensemble wins
        ensemble_wins = combined[combined["ensemble_better"]]
        if len(ensemble_wins) > 0:
            print("\nWhen ensemble wins:")
            print(f"  Mean ensemble error: {ensemble_wins['ensemble_abs_err'].mean():.3f}")
            print(f"  Mean best model error: {ensemble_wins['best_model_abs_err'].mean():.3f}")

        # When best model wins
        best_wins = combined[~combined["ensemble_better"]]
        if len(best_wins) > 0:
            print("\nWhen best model wins:")
            print(f"  Mean ensemble error: {best_wins['ensemble_abs_err'].mean():.3f}")
            print(f"  Mean best model error: {best_wins['best_model_abs_err'].mean():.3f}")

        return {
            "ensemble_mae": ensemble_mae,
            "best_strategy": best_strategy,
            "best_model": best_model,
            "best_model_mae": best_model_mae,
            "ensemble_win_rate": n_better / n_total,
        }


if __name__ == "__main__":
    # Example usage

    print("\n" + "*" * 80)
    print("Full ensemble analysis: XGB + PyTorch + ChemProp")
    print("*" * 80)
    sim = MetaModelSimulator(
        ["logd-reg-xgb", "logd-reg-pytorch", "logd-reg-chemprop"],
        id_column="molecule_name",
    )
    sim.report(details=True)  # Full analysis

    print("\n" + "*" * 80)
    print("Two model ensemble analysis: PyTorch + ChemProp")
    print("*" * 80)
    sim = MetaModelSimulator(
        ["logd-reg-pytorch", "logd-reg-chemprop"],
        id_column="molecule_name",
    )
    sim.report(details=True)  # Full analysis
