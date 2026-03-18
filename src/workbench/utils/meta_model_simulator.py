"""MetaModelSimulator: Simulate and analyze ensemble model performance.

This class helps evaluate whether a meta model (ensemble) would outperform
individual child models by analyzing endpoint inference predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

from workbench.api import Model
from workbench.model_script_utils.meta_model_utils import conf_weights_with_fallback, ensemble_confidence

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

    def __init__(self, model_names: list[str], id_column: str = "id", capture_name: str = "full_cross_fold"):
        """Initialize the simulator with a list of model names.

        Args:
            model_names: List of model names to include in the ensemble
            id_column: Column name to use for row alignment (default: "id")
            capture_name: Inference capture name to load predictions from (default: "full_cross_fold")
        """
        self.model_names = model_names
        self.id_column = id_column
        self.capture_name = capture_name
        self._dfs: dict[str, pd.DataFrame] = {}
        self._conf_error_corr: dict[str, float] = {}
        self._target_column: str | None = None
        self._load_predictions()

    def _load_predictions(self):
        """Load endpoint inference predictions for all models."""
        log.info(f"Loading predictions for {len(self.model_names)} models...")
        for name in self.model_names:
            model = Model(name)
            if self._target_column is None:
                self._target_column = model.target()
            df = model.get_inference_predictions(self.capture_name)
            if df is None:
                raise ValueError(
                    f"No '{self.capture_name}' predictions found for model '{name}'. Run endpoint inference first."
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

        # Compute confidence-to-error correlation on aligned data
        for name, df in self._dfs.items():
            if "confidence" in df.columns:
                self._conf_error_corr[name] = stats.spearmanr(df["confidence"], df["abs_residual"])[0]
            else:
                self._conf_error_corr[name] = 0.0

    def reproduce_deployed(
        self,
        aggregation_strategy: str,
        model_weights: dict[str, float],
        corr_scale: dict[str, float] | None = None,
        optimal_alpha: float = 0.5,
        endpoint_to_model: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Reproduce the deployed meta endpoint's aggregation logic exactly.

        Uses the same algorithm as the template's aggregate_predictions() so that
        results can be compared 1:1 with actual endpoint output.

        Args:
            aggregation_strategy (str): Strategy name (e.g. 'inverse_mae_weighted')
            model_weights (dict): Endpoint-name -> weight mapping from meta config.
                If endpoint_to_model is provided, keys are endpoint names that get
                mapped to model names; otherwise keys must be model names.
            corr_scale (dict): Endpoint-name -> |conf_error_corr| mapping
            optimal_alpha (float): Blend weight for ensemble confidence
            endpoint_to_model (dict): Optional endpoint-name -> model-name mapping.
                When provided, model_weights/corr_scale keys are treated as endpoint
                names and translated to model names for lookup.

        Returns:
            pd.DataFrame: DataFrame with id, target, prediction, prediction_std,
                confidence columns — matching the template output format
        """
        model_names = list(self._dfs.keys())

        # Map endpoint-keyed dicts to model-keyed dicts if needed
        if endpoint_to_model:
            model_to_ep = {m: ep for ep, m in endpoint_to_model.items()}
            mw = {m: model_weights.get(model_to_ep.get(m, m), 1.0) for m in model_names}
            cs = {m: (corr_scale or {}).get(model_to_ep.get(m, m), 1.0) for m in model_names}
        else:
            mw = {m: model_weights.get(m, 1.0) for m in model_names}
            cs = {m: (corr_scale or {}).get(m, 1.0) for m in model_names}

        # Build arrays (same order as model_names)
        pred_arr = np.column_stack([self._dfs[name]["prediction"].values for name in model_names])
        conf_arr = np.column_stack([self._dfs[name]["confidence"].values for name in model_names])
        target = self._dfs[model_names[0]][self._target_column].values
        ids = self._dfs[model_names[0]][self.id_column].values

        # Fallback weights (normalized), matching template logic
        fallback_w = np.array([mw[name] for name in model_names])
        fallback_w = fallback_w / fallback_w.sum()

        # Compute per-row weights — exactly mirroring template's aggregate_predictions()
        if aggregation_strategy == "simple_mean":
            row_weights = np.ones_like(pred_arr) / len(model_names)

        elif aggregation_strategy == "confidence_weighted":
            row_weights = conf_weights_with_fallback(conf_arr, fallback_w)

        elif aggregation_strategy == "inverse_mae_weighted":
            row_weights = np.broadcast_to(fallback_w, pred_arr.shape)

        elif aggregation_strategy == "scaled_conf_weighted":
            row_weights = conf_weights_with_fallback(conf_arr * fallback_w, fallback_w)

        elif aggregation_strategy == "calibrated_conf_weighted":
            scale = np.array([cs[name] for name in model_names])
            row_weights = conf_weights_with_fallback(conf_arr * scale, fallback_w)

        else:
            raise ValueError(f"Unknown aggregation_strategy: {aggregation_strategy}")

        # Weighted prediction
        prediction = (pred_arr * row_weights).sum(axis=1)

        # Ensemble std across endpoints
        pred_std = pd.DataFrame({name: self._dfs[name]["prediction"].values for name in model_names}).std(axis=1).values

        # Ensemble confidence (matches deployed template)
        cs_arr = np.array([cs[name] for name in model_names])
        confidence = ensemble_confidence(pred_arr, conf_arr, cs_arr, fallback_w, optimal_alpha)

        return pd.DataFrame(
            {
                self.id_column: ids,
                self._target_column: target,
                "prediction": prediction,
                "prediction_std": pred_std,
                "confidence": confidence,
            }
        )

    def report(self, details: bool = False):
        """Print a comprehensive analysis report

        Args:
            details: Whether to include detailed sections (default: False)
        """
        self.model_performance()
        self.residual_correlations()
        self.strategy_comparison()
        self.ensemble_confidence_analysis()
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
                    "conf_err_corr": self._conf_error_corr[name],
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
        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()

        weights = conf_weights_with_fallback(conf_arr, inv_mae_weights)
        combined["conf_weighted"] = (pred_arr * weights).sum(axis=1)
        mae = (combined["conf_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Confidence-Weighted", "mae": mae})

        # Strategy 3: Inverse-MAE weighted
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
        scaled_weights = conf_weights_with_fallback(scaled_conf, inv_mae_weights)
        combined["scaled_conf_weighted"] = (pred_arr * scaled_weights).sum(axis=1)
        mae = (combined["scaled_conf_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Scaled Conf-Weighted", "mae": mae})

        # Strategy 6: Calibrated confidence (confidence scaled by |confidence_to_error_corr|)
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])
        cal_conf = conf_arr * corr_scale
        cal_weights = conf_weights_with_fallback(cal_conf, inv_mae_weights)
        combined["cal_conf_weighted"] = (pred_arr * cal_weights).sum(axis=1)
        mae = (combined["cal_conf_weighted"] - combined["target"]).abs().mean()
        results.append({"strategy": "Calibrated Conf-Weighted", "mae": mae})

        # Strategy 7: Drop worst model (use simple mean of remaining, or raw prediction if only 1 left)
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

    def ensemble_confidence_analysis(self) -> dict:
        """Analyze ensemble confidence by blending model agreement with calibrated confidence.

        Uses the same confidence formula as the deployed template:
          confidence = alpha * agreement + (1 - alpha) * cal_conf
        where:
          - agreement = 1 / (1 + pred_std)
          - cal_conf = (conf * corr_scale * model_weights).sum(axis=1)

        Grid searches alpha to find the optimal blend, then reports all variants.

        Returns:
            Dict with ensemble confidence results including optimal alpha and correlations
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE CONFIDENCE ANALYSIS")
        print("=" * 60)

        model_names = list(self._dfs.keys())

        # Build combined arrays
        pred_arr = np.column_stack([self._dfs[name]["prediction"].values for name in model_names])
        conf_arr = np.column_stack([self._dfs[name]["confidence"].values for name in model_names])

        # Ensemble prediction (simple mean) and its absolute residual
        target = self._dfs[model_names[0]][self._target_column].values
        ensemble_pred = pred_arr.mean(axis=1)
        ensemble_abs_err = np.abs(ensemble_pred - target)

        # Compute model weights and corr_scale
        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])

        # Report individual model baselines
        print("\nIndividual model confidence-to-error correlations:")
        for name in model_names:
            print(f"  {name}: {self._conf_error_corr[name]:.3f}")

        # Report agreement-only and calibrated-conf-only
        agreement_only = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, 1.0)
        cal_conf_only = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, 0.0)
        corr_agreement = stats.spearmanr(agreement_only, ensemble_abs_err)[0]
        corr_cal_conf = stats.spearmanr(cal_conf_only, ensemble_abs_err)[0]
        print(f"\nAgreement-only          (alpha=1.0): conf_error_corr = {corr_agreement:.3f}")
        print(f"Calibrated-conf-only    (alpha=0.0): conf_error_corr = {corr_cal_conf:.3f}")

        # Grid search alpha
        best_alpha = 0.0
        best_corr = corr_cal_conf
        alpha_results = []
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, alpha)
            corr = stats.spearmanr(blended, ensemble_abs_err)[0]
            alpha_results.append({"alpha": alpha, "conf_error_corr": corr})
            if corr < best_corr:  # More negative = better
                best_corr = corr
                best_alpha = alpha

        print(f"Optimal blend           (alpha={best_alpha:.2f}): conf_error_corr = {best_corr:.3f}")

        # Show the full alpha sweep
        print("\nAlpha sweep (alpha=1 → agreement only, alpha=0 → calibrated conf only):")
        for r in alpha_results:
            marker = " <-- best" if abs(r["alpha"] - best_alpha) < 0.01 else ""
            print(f"  alpha={r['alpha']:.2f}: {r['conf_error_corr']:.3f}{marker}")

        return {
            "agreement_corr": corr_agreement,
            "calibrated_conf_corr": corr_cal_conf,
            "best_alpha": best_alpha,
            "best_blend_corr": best_corr,
            "alpha_sweep": alpha_results,
        }

    def best_ensemble_predictions(self) -> pd.DataFrame:
        """Generate predictions DataFrame for the best ensemble strategy with blended confidence.

        Uses the same confidence formula as the deployed template:
          confidence = alpha * agreement + (1 - alpha) * cal_conf
        where:
          - agreement = 1 / (1 + pred_std)
          - cal_conf = (conf * corr_scale * model_weights).sum(axis=1)

        Returns a DataFrame matching the format of individual model predictions:
        id_column, target, prediction, confidence, residual, abs_residual.

        Returns:
            pd.DataFrame with ensemble predictions and confidence
        """
        model_names = list(self._dfs.keys())

        # Build combined arrays
        pred_arr = np.column_stack([self._dfs[name]["prediction"].values for name in model_names])
        conf_arr = np.column_stack([self._dfs[name]["confidence"].values for name in model_names])
        target = self._dfs[model_names[0]][self._target_column].values
        ids = self._dfs[model_names[0]][self.id_column].values

        # Find best strategy (replicate logic from strategy_comparison)
        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])

        strategies = {
            "Simple Mean": pred_arr.mean(axis=1),
            "Inverse-MAE Weighted": (pred_arr * inv_mae_weights).sum(axis=1),
        }

        # Confidence-weighted (fallback to inv_mae_weights when all confs are 0)
        weights = conf_weights_with_fallback(conf_arr, inv_mae_weights)
        strategies["Confidence-Weighted"] = (pred_arr * weights).sum(axis=1)

        # Scaled conf-weighted
        scaled_conf = conf_arr * inv_mae_weights
        scaled_weights = conf_weights_with_fallback(scaled_conf, inv_mae_weights)
        strategies["Scaled Conf-Weighted"] = (pred_arr * scaled_weights).sum(axis=1)

        # Calibrated conf-weighted
        cal_conf_weights = conf_arr * corr_scale
        cal_weights = conf_weights_with_fallback(cal_conf_weights, inv_mae_weights)
        strategies["Calibrated Conf-Weighted"] = (pred_arr * cal_weights).sum(axis=1)

        # Drop worst
        worst_model = max(mae_scores, key=mae_scores.get)
        remaining_idx = [i for i, n in enumerate(model_names) if n != worst_model]
        if len(remaining_idx) > 1:
            strategies[f"Drop Worst ({worst_model})"] = pred_arr[:, remaining_idx].mean(axis=1)

        # Select best strategy by MAE
        strategy_maes = {name: np.abs(preds - target).mean() for name, preds in strategies.items()}
        best_strategy = min(strategy_maes, key=strategy_maes.get)
        best_pred = strategies[best_strategy]

        # Find optimal alpha for ensemble confidence
        ensemble_abs_err = np.abs(best_pred - target)
        best_alpha, best_corr = (
            0.0,
            stats.spearmanr(
                ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, 0.0), ensemble_abs_err
            )[0],
        )
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, alpha)
            corr = stats.spearmanr(blended, ensemble_abs_err)[0]
            if corr < best_corr:
                best_corr = corr
                best_alpha = alpha

        confidence = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, best_alpha)

        # Build output DataFrame
        result = pd.DataFrame(
            {
                self.id_column: ids,
                self._target_column: target,
                "prediction": best_pred,
                "confidence": confidence,
                "residual": best_pred - target,
                "abs_residual": np.abs(best_pred - target),
            }
        )

        print(f"\nBest ensemble: {best_strategy} (MAE={strategy_maes[best_strategy]:.4f})")
        print(f"Ensemble confidence: alpha={best_alpha:.2f}, conf_error_corr={best_corr:.3f}")

        return result

    def get_best_strategy_config(self) -> dict:
        """Get the best ensemble strategy configuration for MetaModel creation.

        Evaluates all strategies, picks the best one by MAE, and returns the
        template parameters needed for MetaModel.create().

        If "Drop Worst" wins, the worst model is excluded from endpoints
        and the remaining strategies are re-evaluated on the reduced set.

        Returns:
            Dict with keys: aggregation_strategy, model_weights, corr_scale,
            endpoints, target_column
        """
        model_names = list(self._dfs.keys())
        config = self._compute_strategy_config(model_names)

        # If drop_worst won, re-evaluate with reduced model set
        if config["aggregation_strategy"] == "drop_worst":
            mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
            worst_model = max(mae_scores, key=mae_scores.get)
            remaining = [n for n in model_names if n != worst_model]
            log.info(f"Drop Worst won: excluding '{worst_model}', re-evaluating with {remaining}")
            config = self._compute_strategy_config(remaining)

        log.info(f"Best strategy config: {config['aggregation_strategy']}")
        return config

    def _compute_strategy_config(self, model_names: list[str]) -> dict:
        """Compute the best strategy and its config for a given set of models.

        Args:
            model_names: List of model names to evaluate

        Returns:
            Dict with strategy configuration
        """
        # Build combined arrays
        pred_arr = np.column_stack([self._dfs[name]["prediction"].values for name in model_names])
        conf_arr = np.column_stack([self._dfs[name]["confidence"].values for name in model_names])
        target = self._dfs[model_names[0]][self._target_column].values

        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])

        # Evaluate all strategies (confidence strategies use fallback to inv_mae_weights when all confs are 0)
        strategies = {}
        strategies["simple_mean"] = pred_arr.mean(axis=1)

        weights = conf_weights_with_fallback(conf_arr, inv_mae_weights)
        strategies["confidence_weighted"] = (pred_arr * weights).sum(axis=1)

        strategies["inverse_mae_weighted"] = (pred_arr * inv_mae_weights).sum(axis=1)

        scaled_conf = conf_arr * inv_mae_weights
        scaled_weights = conf_weights_with_fallback(scaled_conf, inv_mae_weights)
        strategies["scaled_conf_weighted"] = (pred_arr * scaled_weights).sum(axis=1)

        cal_conf = conf_arr * corr_scale
        cal_weights = conf_weights_with_fallback(cal_conf, inv_mae_weights)
        strategies["calibrated_conf_weighted"] = (pred_arr * cal_weights).sum(axis=1)

        # Drop worst (only if > 2 models)
        if len(model_names) > 2:
            worst_model = max(mae_scores, key=mae_scores.get)
            remaining_idx = [i for i, n in enumerate(model_names) if n != worst_model]
            strategies["drop_worst"] = pred_arr[:, remaining_idx].mean(axis=1)

        # Find best by MAE
        strategy_maes = {name: np.abs(preds - target).mean() for name, preds in strategies.items()}
        best_strategy = min(strategy_maes, key=strategy_maes.get)

        # Compute optimal_alpha for ensemble confidence blending
        best_pred = strategies[best_strategy]
        ensemble_abs_err = np.abs(best_pred - target)

        best_alpha, best_corr = (
            0.0,
            stats.spearmanr(
                ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, 0.0), ensemble_abs_err
            )[0],
        )
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, alpha)
            corr = stats.spearmanr(blended, ensemble_abs_err)[0]
            if corr < best_corr:  # More negative = better
                best_corr = corr
                best_alpha = alpha

        log.info(f"Optimal alpha for ensemble confidence: {best_alpha:.2f} (conf_error_corr={best_corr:.3f})")

        # Build config dict
        model_weights_dict = {name: float(w) for name, w in zip(model_names, inv_mae_weights)}
        corr_scale_dict = {name: float(c) for name, c in zip(model_names, corr_scale)}

        return {
            "aggregation_strategy": best_strategy,
            "model_weights": model_weights_dict,
            "corr_scale": corr_scale_dict,
            "optimal_alpha": float(best_alpha),
            "endpoints": model_names,
            "target_column": self._target_column,
        }

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
        weights = conf_weights_with_fallback(conf_arr, inv_mae_weights)
        ensemble_strategies["Confidence-Weighted"] = (pred_arr * weights).sum(axis=1)
        ensemble_strategies["Inverse-MAE Weighted"] = (pred_arr * inv_mae_weights).sum(axis=1)
        scaled_conf = conf_arr * inv_mae_weights
        scaled_weights = conf_weights_with_fallback(scaled_conf, inv_mae_weights)
        ensemble_strategies["Scaled Conf-Weighted"] = (pred_arr * scaled_weights).sum(axis=1)
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])
        cal_conf = conf_arr * corr_scale
        cal_weights = conf_weights_with_fallback(cal_conf, inv_mae_weights)
        ensemble_strategies["Calibrated Conf-Weighted"] = (pred_arr * cal_weights).sum(axis=1)
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
