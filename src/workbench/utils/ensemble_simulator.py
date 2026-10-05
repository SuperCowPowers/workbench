"""EnsembleSimulator: Simulate and analyze ensemble model performance.

This class helps evaluate whether an ensemble of models would outperform
individual child models by analyzing endpoint inference predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

from workbench.api import Model
from workbench.utils.ensemble_utils import conf_weights_with_fallback, ensemble_confidence

# Set up the log
log = logging.getLogger("workbench")


class EnsembleSimulator:
    """Simulate ensemble performance from child model predictions.

    This class loads cross-validation predictions from multiple models and
    analyzes how different ensemble strategies would perform compared to
    the individual models.

    Example:
        ```python
        from workbench.utils.ensemble_simulator import EnsembleSimulator

        sim = EnsembleSimulator(["model-a", "model-b", "model-c"])
        sim.report()  # Print full analysis
        sim.strategy_comparison()  # Compare ensemble strategies
        ```
    """

    def __init__(
        self,
        model_names: list[str],
        id_column: str = "id",
        capture_name: str | None = None,
        target: str | None = None,
    ):
        """Initialize the simulator with a list of model names.

        A multi-target model keeps one out-of-fold capture per target (`cv_<target>`),
        already remapped so `prediction` / `prediction_std` / `confidence` carry that
        target's values. Analysis is therefore per target: member decorrelation and
        weights on one target say nothing about another. Build one simulator per target
        you care about:

            {t: EnsembleSimulator(models, target=t) for t in targets}

        Args:
            model_names: List of model names to include in the ensemble
            id_column: Column name to use for row alignment (default: "id")
            capture_name: Inference capture to load. Defaults to the one holding `target`'s
                out-of-fold rows — `full_cross_fold` single-target, `cv_<target>` multi.
            target: Target column to analyze. Required for multi-target models.
        """
        self.model_names = model_names
        self.id_column = id_column
        self.capture_name = capture_name
        self._requested_capture = capture_name
        self._requested_target = target
        # Resolved per model, not once: a pool can mix a multi-target model (cv_<target>)
        # with a single-target one trained on that same target (full_cross_fold).
        self.capture_names: dict[str, str] = {}
        self._dfs: dict[str, pd.DataFrame] = {}
        self._conf_error_corr: dict[str, float] = {}
        self._target_column: str | None = None
        self._load_predictions()

    #: Score columns and whether a larger value is better. MAE answers placement; the two
    #: correlations answer ordering, which is what survives a downstream recalibration.
    METRICS = {"mae": False, "spearman": True, "pearson": True}

    @classmethod
    def _score_strategies(cls, strategies: dict[str, np.ndarray], target: np.ndarray) -> pd.DataFrame:
        """Score each strategy's prediction vector on every metric in `METRICS`.

        Returns a frame indexed by strategy name with one column per metric.
        """
        target = np.asarray(target, dtype=float)
        rows = {}
        for name, preds in strategies.items():
            preds = np.asarray(preds, dtype=float)
            both = np.isfinite(preds) & np.isfinite(target)
            rows[name] = {
                "mae": float(np.abs(preds[both] - target[both]).mean()),
                "spearman": cls._rank_corr(preds, target, f"{name} vs target"),
                "pearson": float(stats.pearsonr(preds[both], target[both])[0]),
            }
        return pd.DataFrame(rows).T[list(cls.METRICS)]

    @classmethod
    def _best_strategy(cls, scores: pd.DataFrame, select_by: str) -> str:
        """The winning strategy name under `select_by`."""
        if select_by not in cls.METRICS:
            raise ValueError(f"select_by must be one of {list(cls.METRICS)}, got {select_by!r}")
        column = scores[select_by]
        return str(column.idxmax() if cls.METRICS[select_by] else column.idxmin())

    @staticmethod
    def _rank_corr(x, y, label: str, min_pairs: int = 30, warn: bool = True) -> float:
        """Spearman correlation over the rows where both inputs are finite.

        A member whose confidence is missing on some rows still carries honest confidence
        on the rest, and SciPy's default policy would propagate a single NaN into the
        correlation. That NaN reaches `corr_scale`, where it makes every calibrated weight
        fall back to static and is written into a deployed aggregation node — whose
        `ensemble_confidence` has no NaN guard and would serve NaN confidence.

        Returns 0.0 below `min_pairs` finite pairs, which disables calibration for that
        member rather than scaling it by a number estimated from almost nothing.

        Args:
            warn: Log the dropped-row count. False inside a sweep, where every step drops
                the same rows and the caller reports the coverage once.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        both = np.isfinite(x) & np.isfinite(y)
        n = int(both.sum())
        if warn and n < len(x):
            log.warning(f"{label}: correlation over {n:,} of {len(x):,} rows; the rest are missing a value")
        if n < min_pairs:
            log.warning(f"{label}: only {n:,} finite pairs (min {min_pairs}), reporting 0.0")
            return 0.0
        corr = stats.spearmanr(x[both], y[both])[0]
        return 0.0 if not np.isfinite(corr) else float(corr)

    @staticmethod
    def _declared_targets(model: Model) -> list[str]:
        """A model's targets as a list, however it declares them.

        `target()` returns a str or a list, and a list of one is a single-target model —
        `EndpointCore` gates on `len(targets) > 1`, so a one-element list captures under
        `full_cross_fold` like any other single-target model. Matching that test here is
        what keeps capture resolution agreeing with what was actually written.
        """
        declared = model.target()
        return list(declared) if isinstance(declared, list) else [declared]

    @classmethod
    def _resolve_target(cls, model: Model, target: str | None) -> str:
        """The target column to analyze, checked against what the model declares."""
        declared = cls._declared_targets(model)
        if len(declared) > 1:
            if target is None:
                raise ValueError(f"Model '{model.name}' is multi-target — pass target= to choose one of: {declared}")
            if target not in declared:
                raise ValueError(f"Target {target!r} not among model '{model.name}' targets: {declared}")
            return target
        if target is not None and target != declared[0]:
            raise ValueError(f"Model '{model.name}' has a single target {declared[0]!r}, not {target!r}")
        return declared[0]

    def _load_predictions(self):
        """Load endpoint inference predictions for all models."""
        log.info(f"Loading predictions for {len(self.model_names)} models...")
        for name in self.model_names:
            model = Model(name)
            target = self._resolve_target(model, self._requested_target)
            if self._target_column is None:
                self._target_column = target
            elif target != self._target_column:
                raise ValueError(f"Model '{name}' resolves to target {target!r}, not {self._target_column!r}")
            multi = len(self._declared_targets(model)) > 1
            capture = self._requested_capture or (f"cv_{target}" if multi else "full_cross_fold")
            self.capture_names[name] = capture
            df = model.get_inference_predictions(capture)
            if df is None:
                raise ValueError(f"No '{capture}' predictions found for model '{name}'. Run endpoint inference first.")
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
                self._conf_error_corr[name] = self._rank_corr(
                    df["confidence"], df["abs_residual"], f"{name} conf-to-error"
                )
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

    def report(self, details: bool = False, select_by: str = "mae"):
        """Print a comprehensive analysis report

        Args:
            details: Whether to include detailed sections (default: False)
            select_by: Metric deciding which strategy wins -- "mae", "spearman" or "pearson"
        """
        self.model_performance()
        self.residual_correlations()
        self.strategy_comparison(select_by=select_by)
        self.ensemble_confidence_analysis()
        self.ensemble_failure_analysis(select_by=select_by)
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
            n_missing = int(conf.isna().sum())
            print(
                f"  Confidence: mean={conf.mean():.3f}, std={conf.std():.3f}, "
                f"min={conf.min():.3f}, max={conf.max():.3f}, missing={n_missing:,}/{len(conf):,}"
            )

            # Correlations over the rows carrying both values; a single NaN would propagate.
            both = conf.notna() & df["abs_residual"].notna()
            corr_pearson, p_pearson = stats.pearsonr(conf[both], df["abs_residual"][both])
            corr_spearman, p_spearman = stats.spearmanr(conf[both], df["abs_residual"][both])

            print(f"  Confidence vs |residual| over {int(both.sum()):,} rows:")
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

    def _build_strategies(self, model_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
        """Every candidate strategy's prediction vector, over `model_names`.

        Returns `(strategies, target, context)`, where context carries the per-model
        arrays the caller needs to describe the winner: `inv_mae_weights`, `corr_scale`,
        `pred_arr`, `conf_arr`, `mae_scores` and `worst_model`.
        """
        pred_arr = np.column_stack([self._dfs[name]["prediction"].values for name in model_names])
        conf_arr = np.column_stack([self._dfs[name]["confidence"].values for name in model_names])
        target = self._dfs[model_names[0]][self._target_column].values

        mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
        inv_mae_weights = np.array([1.0 / mae_scores[name] for name in model_names])
        inv_mae_weights = inv_mae_weights / inv_mae_weights.sum()
        corr_scale = np.array([abs(self._conf_error_corr[name]) for name in model_names])

        strategies = {"simple_mean": pred_arr.mean(axis=1), "inverse_mae_weighted": pred_arr @ inv_mae_weights}
        for key, conf in (
            ("confidence_weighted", conf_arr),
            ("scaled_conf_weighted", conf_arr * inv_mae_weights),
            ("calibrated_conf_weighted", conf_arr * corr_scale),
        ):
            weights = conf_weights_with_fallback(conf, inv_mae_weights)
            strategies[key] = (pred_arr * weights).sum(axis=1)

        worst_model = max(mae_scores, key=mae_scores.get)
        best_model = min(mae_scores, key=mae_scores.get)
        strategies["best_model_only"] = pred_arr[:, model_names.index(best_model)]
        if len(model_names) > 2:
            remaining = [i for i, n in enumerate(model_names) if n != worst_model]
            strategies["drop_worst"] = pred_arr[:, remaining].mean(axis=1)

        context = {
            "pred_arr": pred_arr,
            "conf_arr": conf_arr,
            "inv_mae_weights": inv_mae_weights,
            "corr_scale": corr_scale,
            "mae_scores": mae_scores,
            "worst_model": worst_model,
            "best_model": best_model,
        }
        return strategies, target, context

    def strategy_comparison(self, select_by: str = "mae") -> pd.DataFrame:
        """Compare ensemble strategies on placement and on ordering.

        MAE scores placement, which a downstream recalibration can re-derive; Spearman and
        Pearson score ordering, which it cannot. Read all three before picking: a strategy
        that averages toward the mean wins MAE while packing near-ties that cost rank.

        Args:
            select_by: Metric to sort by and mark as the winner -- "mae", "spearman" or "pearson"

        Returns:
            DataFrame indexed by strategy with an mae, spearman and pearson column, best first
        """
        print("\n" + "=" * 60)
        print(f"ENSEMBLE STRATEGY COMPARISON (by {select_by})")
        print("=" * 60)

        model_names = list(self._dfs.keys())
        strategies, target, context = self._build_strategies(model_names)
        scores = self._score_strategies(strategies, target)
        scores = scores.sort_values(select_by, ascending=not self.METRICS[select_by])

        print("\n" + scores.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\nWinner by {select_by}: {self._best_strategy(scores, select_by)}")
        print(f"  best member: {context['best_model']}, worst: {context['worst_model']}")

        print("\nIndividual models for reference:")
        members = self._score_strategies(
            {name: self._dfs[name]["prediction"].values for name in model_names}, target
        ).sort_values(select_by, ascending=not self.METRICS[select_by])
        print(members.to_string(float_format=lambda v: f"{v:.4f}"))

        return scores

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
        corr_agreement = self._rank_corr(agreement_only, ensemble_abs_err, "agreement-only confidence")
        corr_cal_conf = self._rank_corr(cal_conf_only, ensemble_abs_err, "calibrated-conf-only confidence")
        print(f"\nAgreement-only          (alpha=1.0): conf_error_corr = {corr_agreement:.3f}")
        print(f"Calibrated-conf-only    (alpha=0.0): conf_error_corr = {corr_cal_conf:.3f}")

        # Grid search alpha
        best_alpha = 0.0
        best_corr = corr_cal_conf
        alpha_results = []
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = ensemble_confidence(pred_arr, conf_arr, corr_scale, inv_mae_weights, alpha)
            corr = self._rank_corr(blended, ensemble_abs_err, "ensemble confidence", warn=False)
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

    def best_ensemble_predictions(self, select_by: str = "mae") -> pd.DataFrame:
        """Predictions for the winning ensemble strategy, with blended confidence.

        Uses the same confidence formula as the deployed template:
          confidence = alpha * agreement + (1 - alpha) * cal_conf
        where:
          - agreement = 1 / (1 + pred_std)
          - cal_conf = (conf * corr_scale * model_weights).sum(axis=1)

        Args:
            select_by: Metric deciding the winner -- "mae", "spearman" or "pearson"

        Returns:
            DataFrame matching the individual-model format: id_column, target, prediction,
            confidence, residual, abs_residual
        """
        model_names = list(self._dfs.keys())
        strategies, target, ctx = self._build_strategies(model_names)
        ids = self._dfs[model_names[0]][self.id_column].values

        scores = self._score_strategies(strategies, target)
        best_strategy = self._best_strategy(scores, select_by)
        best_pred = strategies[best_strategy]

        best_alpha, best_corr = self._optimal_alpha(ctx, np.abs(best_pred - target))
        confidence = ensemble_confidence(
            ctx["pred_arr"], ctx["conf_arr"], ctx["corr_scale"], ctx["inv_mae_weights"], best_alpha
        )

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

        print(f"\nBest ensemble by {select_by}: {best_strategy}")
        print("  " + "  ".join(f"{m}={scores.loc[best_strategy, m]:.4f}" for m in self.METRICS))
        print(f"Ensemble confidence: alpha={best_alpha:.2f}, conf_error_corr={best_corr:.3f}")

        return result

    def _optimal_alpha(self, ctx: dict, ensemble_abs_err: np.ndarray) -> tuple[float, float]:
        """The agreement/calibrated-confidence blend whose confidence best tracks error.

        More negative is better: confidence should fall as error rises.
        """
        best_alpha, best_corr = 0.0, np.inf
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = ensemble_confidence(
                ctx["pred_arr"], ctx["conf_arr"], ctx["corr_scale"], ctx["inv_mae_weights"], alpha
            )
            corr = self._rank_corr(blended, ensemble_abs_err, "ensemble confidence", warn=False)
            if corr < best_corr:
                best_corr = corr
                best_alpha = float(alpha)
        return best_alpha, best_corr

    def get_best_strategy_config(self, select_by: str = "mae") -> dict:
        """The winning strategy and the parameters an aggregation node needs to build it.

        If "drop_worst" wins, the worst member is excluded from endpoints and the remaining
        strategies are re-evaluated on the reduced set.

        Args:
            select_by: Metric deciding the winner -- "mae", "spearman" or "pearson".
                Ordering metrics are the ones to use when a recalibration follows.

        Returns:
            Dict with keys: aggregation_strategy, model_weights, corr_scale, optimal_alpha,
            endpoints, target_column, scores
        """
        model_names = list(self._dfs.keys())
        config = self._compute_strategy_config(model_names, select_by)

        if config["aggregation_strategy"] == "drop_worst":
            mae_scores = {name: self._dfs[name]["abs_residual"].mean() for name in model_names}
            worst_model = max(mae_scores, key=mae_scores.get)
            remaining = [n for n in model_names if n != worst_model]
            log.info(f"Drop Worst won: excluding '{worst_model}', re-evaluating with {remaining}")
            config = self._compute_strategy_config(remaining, select_by)

        log.info(f"Best strategy config by {select_by}: {config['aggregation_strategy']}")
        return config

    def _compute_strategy_config(self, model_names: list[str], select_by: str = "mae") -> dict:
        """Compute the winning strategy and its config for a given set of models.

        Args:
            model_names: Models to evaluate
            select_by: Metric deciding the winner

        Returns:
            Dict with the strategy configuration, carrying every metric's score so a caller
            can see what the choice cost on the metrics it did not select by
        """
        strategies, target, ctx = self._build_strategies(model_names)
        scores = self._score_strategies(strategies, target)
        best_strategy = self._best_strategy(scores, select_by)

        best_alpha, best_corr = self._optimal_alpha(ctx, np.abs(strategies[best_strategy] - target))
        log.info(f"Optimal alpha for ensemble confidence: {best_alpha:.2f} (conf_error_corr={best_corr:.3f})")

        return {
            "aggregation_strategy": best_strategy,
            "model_weights": {n: float(w) for n, w in zip(model_names, ctx["inv_mae_weights"])},
            "corr_scale": {n: float(c) for n, c in zip(model_names, ctx["corr_scale"])},
            "optimal_alpha": best_alpha,
            "endpoints": model_names,
            "target_column": self._target_column,
            "scores": scores.loc[best_strategy].to_dict(),
        }

    def ensemble_failure_analysis(self, select_by: str = "mae") -> dict:
        """Compare the winning ensemble against the best individual member.

        Scores only strategies that actually combine members, so "best model only" is not
        a candidate -- the comparison would otherwise be against itself.

        Args:
            select_by: Metric deciding the winning ensemble

        Returns:
            Dict with comparison statistics
        """
        print("\n" + "=" * 60)
        print("BEST ENSEMBLE VS BEST MODEL COMPARISON")
        print("=" * 60)

        model_names = list(self._dfs.keys())
        strategies, target, ctx = self._build_strategies(model_names)
        strategies.pop("best_model_only")

        scores = self._score_strategies(strategies, target)
        best_strategy = self._best_strategy(scores, select_by)
        ensemble_pred = strategies[best_strategy]
        ensemble_abs_err = np.abs(ensemble_pred - target)

        best_model = ctx["best_model"]
        best_model_abs_err = self._dfs[best_model]["abs_residual"].values
        best_model_mae = float(ctx["mae_scores"][best_model])
        ensemble_mae = float(scores.loc[best_strategy, "mae"])

        ensemble_better = ensemble_abs_err < best_model_abs_err
        n_better = int(ensemble_better.sum())
        n_total = len(ensemble_abs_err)

        print(f"\nBest individual model: {best_model} (MAE={best_model_mae:.4f})")
        print(f"Best ensemble strategy: {best_strategy}, selected by {select_by}")
        print("  " + "  ".join(f"{m}={scores.loc[best_strategy, m]:.4f}" for m in self.METRICS))
        if ensemble_mae < best_model_mae:
            improvement = (best_model_mae - ensemble_mae) / best_model_mae * 100
            print(f"Ensemble improves over best model by {improvement:.1f}% on MAE")
        else:
            degradation = (ensemble_mae - best_model_mae) / best_model_mae * 100
            print(f"No ensemble benefit on MAE: the best single model is {degradation:.1f}% better")

        print("\nPer-row comparison:")
        print(f"  Ensemble wins: {n_better}/{n_total} ({100*n_better/n_total:.1f}%)")
        print(f"  Best model wins: {n_total - n_better}/{n_total} ({100*(n_total - n_better)/n_total:.1f}%)")

        for label, mask in (("When ensemble wins", ensemble_better), ("When best model wins", ~ensemble_better)):
            if mask.any():
                print(f"\n{label}:")
                print(f"  Mean ensemble error: {ensemble_abs_err[mask].mean():.3f}")
                print(f"  Mean best model error: {best_model_abs_err[mask].mean():.3f}")

        return {
            "ensemble_mae": ensemble_mae,
            "best_strategy": best_strategy,
            "best_model": best_model,
            "best_model_mae": best_model_mae,
            "ensemble_win_rate": n_better / n_total,
            "scores": scores.loc[best_strategy].to_dict(),
        }


if __name__ == "__main__":
    # Example usage

    print("\n" + "*" * 80)
    print("Full ensemble analysis: XGB + PyTorch + ChemProp")
    print("*" * 80)
    sim = EnsembleSimulator(
        ["logd-reg-xgb", "logd-reg-pytorch", "logd-reg-chemprop"],
        id_column="molecule_name",
    )
    sim.report(details=True)  # Full analysis

    print("\n" + "*" * 80)
    print("Two model ensemble analysis: PyTorch + ChemProp")
    print("*" * 80)
    sim = EnsembleSimulator(
        ["logd-reg-pytorch", "logd-reg-chemprop"],
        id_column="molecule_name",
    )
    sim.report(details=True)  # Full analysis
