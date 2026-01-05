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
                raise ValueError(f"No full_inference predictions found for model '{name}'. Run endpoint inference first.")
            df["residual"] = df["prediction"] - df[self._target_column]
            df["abs_residual"] = df["residual"].abs()
            self._dfs[name] = df

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
        """Calculate suggested ensemble weights based on inverse RMSE.

        Returns:
            Dict mapping model name to suggested weight
        """
        print("\n" + "=" * 60)
        print("SUGGESTED ENSEMBLE WEIGHTS")
        print("=" * 60)

        rmse_scores = {name: np.sqrt((df["residual"] ** 2).mean()) for name, df in self._dfs.items()}

        inv_rmse = {name: 1.0 / rmse for name, rmse in rmse_scores.items()}
        total = sum(inv_rmse.values())
        weights = {name: w / total for name, w in inv_rmse.items()}

        print("\nWeights based on inverse RMSE:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f} (RMSE={rmse_scores[name]:.3f})")

        print(f"\nEqual weights would be: {1.0/len(self._dfs):.3f} each")

        return weights

    def strategy_comparison(self) -> pd.DataFrame:
        """Compare different ensemble strategies.

        Returns:
            DataFrame with RMSE for each strategy, sorted best to worst
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
        rmse = np.sqrt(((combined["simple_mean"] - combined["target"]) ** 2).mean())
        results.append({"strategy": "Simple Mean", "rmse": rmse})

        # Strategy 2: Confidence-weighted
        conf_arr = combined[conf_cols].values
        pred_arr = combined[pred_cols].values
        conf_sum = conf_arr.sum(axis=1, keepdims=True) + 1e-8
        weights = conf_arr / conf_sum
        combined["conf_weighted"] = (pred_arr * weights).sum(axis=1)
        rmse = np.sqrt(((combined["conf_weighted"] - combined["target"]) ** 2).mean())
        results.append({"strategy": "Confidence-Weighted", "rmse": rmse})

        # Strategy 3: Inverse-RMSE weighted
        rmse_scores = {name: np.sqrt((self._dfs[name]["residual"] ** 2).mean()) for name in model_names}
        inv_rmse_weights = np.array([1.0 / rmse_scores[name] for name in model_names])
        inv_rmse_weights = inv_rmse_weights / inv_rmse_weights.sum()
        combined["inv_rmse_weighted"] = (pred_arr * inv_rmse_weights).sum(axis=1)
        rmse = np.sqrt(((combined["inv_rmse_weighted"] - combined["target"]) ** 2).mean())
        results.append({"strategy": "Inverse-RMSE Weighted", "rmse": rmse})

        # Strategy 4: Best model only
        best_model = min(rmse_scores, key=rmse_scores.get)
        combined["best_only"] = combined[f"{best_model}_pred"]
        rmse = np.sqrt(((combined["best_only"] - combined["target"]) ** 2).mean())
        results.append({"strategy": f"Best Model Only ({best_model})", "rmse": rmse})

        # Strategy 5: Scaled confidence-weighted
        scaled_conf = conf_arr * inv_rmse_weights
        scaled_conf_sum = scaled_conf.sum(axis=1, keepdims=True) + 1e-8
        scaled_weights = scaled_conf / scaled_conf_sum
        combined["scaled_conf_weighted"] = (pred_arr * scaled_weights).sum(axis=1)
        rmse = np.sqrt(((combined["scaled_conf_weighted"] - combined["target"]) ** 2).mean())
        results.append({"strategy": "Scaled Conf-Weighted", "rmse": rmse})

        # Strategy 6: Drop worst model
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

        results_df = pd.DataFrame(results).sort_values("rmse")
        print("\n" + results_df.to_string(index=False))

        print("\nIndividual model RMSEs for reference:")
        for name, rmse in sorted(rmse_scores.items(), key=lambda x: x[1]):
            print(f"  {name}: {rmse:.4f}")

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
        """Compare ensemble vs best overall model (not per-row oracle).

        Returns:
            Dict with comparison statistics
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE VS BEST MODEL COMPARISON")
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

        conf_cols = [f"{name}_conf" for name in model_names]
        pred_cols = [f"{name}_pred" for name in model_names]

        # Calculate ensemble prediction (confidence-weighted)
        conf_arr = combined[conf_cols].values
        pred_arr = combined[pred_cols].values
        conf_sum = conf_arr.sum(axis=1, keepdims=True) + 1e-8
        weights = conf_arr / conf_sum
        combined["ensemble_pred"] = (pred_arr * weights).sum(axis=1)
        combined["ensemble_abs_err"] = (combined["ensemble_pred"] - combined["target"]).abs()

        # Find best overall model (lowest RMSE)
        rmse_scores = {name: np.sqrt((self._dfs[name]["residual"] ** 2).mean()) for name in model_names}
        best_model = min(rmse_scores, key=rmse_scores.get)
        combined["best_model_abs_err"] = combined[f"{best_model}_abs_err"]

        # Compare ensemble vs best model
        combined["ensemble_better"] = combined["ensemble_abs_err"] < combined["best_model_abs_err"]
        n_better = combined["ensemble_better"].sum()
        n_total = len(combined)

        ensemble_rmse = np.sqrt((combined["ensemble_abs_err"] ** 2).mean())
        best_model_rmse = rmse_scores[best_model]

        print(f"\nBest individual model: {best_model} (RMSE={best_model_rmse:.4f})")
        print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
        if ensemble_rmse < best_model_rmse:
            improvement = (best_model_rmse - ensemble_rmse) / best_model_rmse * 100
            print(f"Ensemble improves over best model by {improvement:.1f}%")
        else:
            degradation = (ensemble_rmse - best_model_rmse) / best_model_rmse * 100
            print(f"Ensemble is worse than best model by {degradation:.1f}%")

        print(f"\nPer-row comparison:")
        print(f"  Ensemble wins: {n_better}/{n_total} ({100*n_better/n_total:.1f}%)")
        print(f"  Best model wins: {n_total - n_better}/{n_total} ({100*(n_total - n_better)/n_total:.1f}%)")

        # When ensemble wins
        ensemble_wins = combined[combined["ensemble_better"]]
        if len(ensemble_wins) > 0:
            print(f"\nWhen ensemble wins:")
            print(f"  Mean ensemble error: {ensemble_wins['ensemble_abs_err'].mean():.3f}")
            print(f"  Mean best model error: {ensemble_wins['best_model_abs_err'].mean():.3f}")

        # When best model wins
        best_wins = combined[~combined["ensemble_better"]]
        if len(best_wins) > 0:
            print(f"\nWhen best model wins:")
            print(f"  Mean ensemble error: {best_wins['ensemble_abs_err'].mean():.3f}")
            print(f"  Mean best model error: {best_wins['best_model_abs_err'].mean():.3f}")

        return {
            "ensemble_rmse": ensemble_rmse,
            "best_model": best_model,
            "best_model_rmse": best_model_rmse,
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
    sim.report()  # Full analysis
    
    print("\n" + "*" * 80)
    print("Two model ensemble analysis: PyTorch + ChemProp")
    print("*" * 80)
    sim = MetaModelSimulator(
        ["logd-reg-pytorch", "logd-reg-chemprop"],
        id_column="molecule_name",
    )
    sim.report()  # Full analysis
