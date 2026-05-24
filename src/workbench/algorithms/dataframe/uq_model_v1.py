"""UQModelV1: conformalized residual-estimator UQ for regression.

Encapsulates the full regression UQ pipeline:
    1. A learned error model (Random Forest) predicting |residual| from
       residual features [prediction, prediction_std, knn_distance,
       knn_target_std, local_pred_gap]
    2. Normalized conformal calibration → prediction intervals with target coverage
    3. Percentile-rank confidence scores

Reference:
    "Uncertainty Quantification in Molecular Machine Learning for Property Predictions
     under Data Shifts" (J Chem Inf Model 2025, PMC12848971) — validates the
     error-model + conformal stack on ADMET endpoints under distribution shift.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Cross-module imports: workbench package path for library use; in-package
# sibling import when symlinked into a script bundle's model_script_utils/ package.
try:
    from workbench.algorithms.dataframe.proximity import Proximity
    from workbench.algorithms.dataframe.residual_features import ResidualFeatures
except ImportError:
    from .proximity import Proximity
    from .residual_features import ResidualFeatures

import logging

log = logging.getLogger("workbench")


# Quantile column names by confidence level (target ± scale × expected_residual)
_QUANTILE_COLUMNS = {
    0.50: ("q_25", "q_75"),
    0.68: ("q_16", "q_84"),
    0.80: ("q_10", "q_90"),
    0.90: ("q_05", "q_95"),
    0.95: ("q_025", "q_975"),
}


class UQModelV1:
    """Conformalized residual-estimator UQ for regression models.

    Usage:
        prox = FingerprintProximity(train_df, id_column="id", target="logp")
        uq = UQModelV1(prox)
        uq.fit(val_df["id"], val_df["logp"], val_df["prediction"], val_df["prediction_std"])
        uq.save(model_dir)

        # Inference:
        uq = UQModelV1.load(model_dir, prox)
        out = uq.predict(test_df[["smiles"]], y_pred, y_pred_std)
        # → DataFrame[expected_residual, confidence, q_025, q_05, ..., q_50, ..., q_975]
    """

    DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]
    FEATURE_ORDER = ["prediction", "prediction_std", "knn_distance", "knn_target_std", "local_pred_gap"]

    def __init__(
        self,
        prox: Proximity,
        confidence_levels: Optional[List[float]] = None,
        k: int = 10,
        training_only_features: bool = True,
        n_estimators: int = 200,
        max_depth: int = 8,
        random_state: int = 42,
    ):
        """
        Args:
            prox: Proximity backend (target required) for neighborhood lookups.
            confidence_levels: Conformal CI levels. Default [0.50, 0.68, 0.80, 0.90, 0.95].
            k: Number of nearest neighbors for residual features. Default 10.
            training_only_features: Whether to restrict neighbors to in_model=True rows
                at training-time feature computation. Default True.
            n_estimators / max_depth / random_state: RandomForestRegressor hyperparameters.
        """
        self.prox = prox
        self.features = ResidualFeatures(prox)
        self.confidence_levels = confidence_levels or list(self.DEFAULT_CONFIDENCE_LEVELS)
        self.k = k
        self.training_only_features = training_only_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Fitted state (set by fit() or load())
        self.error_model: Optional[RandomForestRegressor] = None
        self.scale_factors: Optional[dict] = None
        self.residual_percentiles: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ids: Union[List, pd.Series, np.ndarray],
        y_true: Union[np.ndarray, pd.Series],
        predictions: Union[np.ndarray, pd.Series],
        prediction_std: Union[np.ndarray, pd.Series],
    ) -> "UQModelV1":
        """Fit the error model and conformal calibration on validation predictions.

        Args:
            ids: Validation row IDs (must exist in the proximity reference set).
            y_true: True target values for those rows.
            predictions: Model predictions (ensemble mean).
            prediction_std: Ensemble standard deviation (post log-compression if used upstream).

        Returns:
            self (fitted)
        """
        ids = list(ids) if not isinstance(ids, list) else ids
        y_true = np.asarray(y_true, dtype=float).ravel()
        predictions = np.asarray(predictions, dtype=float).ravel()
        prediction_std = np.asarray(prediction_std, dtype=float).ravel()

        if not (len(ids) == len(y_true) == len(predictions) == len(prediction_std)):
            raise ValueError(
                f"Length mismatch: ids={len(ids)}, y_true={len(y_true)}, "
                f"predictions={len(predictions)}, prediction_std={len(prediction_std)}"
            )

        log.info(f"Fitting UQModelV1 on {len(ids)} validation samples (k={self.k})")

        # 1. Compute neighborhood features
        feat = self.features.compute(
            ids,
            predictions=predictions,
            k=self.k,
            training_only=self.training_only_features,
        )

        X_cal = self._stack_features(predictions, prediction_std, feat)
        y_cal = np.abs(y_true - predictions)

        # 2. Fit error model
        self.error_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.error_model.fit(X_cal, y_cal)

        expected_cal = self.error_model.predict(X_cal)

        # 3. Normalized conformal scale factors
        safe_expected = np.maximum(expected_cal, 1e-10)
        nonconformity = np.abs(y_true - predictions) / safe_expected
        self.scale_factors = {}
        n = len(nonconformity)
        for alpha in self.confidence_levels:
            adjusted_quantile = min(np.ceil((n + 1) * alpha) / n, 1.0)
            self.scale_factors[f"{alpha:.2f}"] = float(np.quantile(nonconformity, adjusted_quantile))

        # 4. Percentile distribution of expected residuals (for confidence ranking)
        self.residual_percentiles = np.asarray([float(np.percentile(expected_cal, p)) for p in range(101)])

        # Diagnostics
        self._log_fit_diagnostics(y_true, predictions, expected_cal)

        return self

    def _log_fit_diagnostics(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        expected_cal: np.ndarray,
    ):
        """Print a fit summary block — coverage, sharpness, feature importance."""
        print("\n" + "=" * 60)
        print("UQModelV1 fit diagnostics")
        print("=" * 60)
        print(f"  Cal samples:           {len(y_true)}")
        print(f"  k (neighbors):         {self.k}")
        print(f"  Mean expected |res|:   {expected_cal.mean():.4f}")
        print(f"  Median expected |res|: {np.median(expected_cal):.4f}")
        print(f"  Mean actual |res|:     {np.mean(np.abs(y_true - predictions)):.4f}")

        # Coverage check
        print("\n  Coverage on cal set:")
        for alpha in self.confidence_levels:
            q = self.scale_factors[f"{alpha:.2f}"]
            lower = predictions - q * expected_cal
            upper = predictions + q * expected_cal
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            mean_width = np.mean(2 * q * expected_cal)
            print(
                f"    {alpha * 100:>5.1f}% CI: scale={q:.3f}, coverage={coverage * 100:5.1f}%, "
                f"mean_width={mean_width:.3f}"
            )

        # Feature importance
        print("\n  Feature importance (Random Forest):")
        for feat_name, importance in zip(self.FEATURE_ORDER, self.error_model.feature_importances_):
            print(f"    {feat_name:<20s} {importance:.4f}")
        print()

    # ------------------------------------------------------------------
    # Inference (auto-dispatch on query type)
    # ------------------------------------------------------------------

    def predict(
        self,
        query: Union[List, pd.Series, np.ndarray, pd.DataFrame],
        predictions: Union[np.ndarray, pd.Series],
        prediction_std: Union[np.ndarray, pd.Series],
    ) -> pd.DataFrame:
        """Compute UQ outputs (expected residual, confidence, intervals) for queries.

        Auto-dispatches on `query` type:
            - list / Series / ndarray of IDs → looks up neighbors by ID
            - DataFrame → novel-query path; must contain backend-specific columns
              (e.g. 'smiles' for FingerprintProximity)

        Args:
            query: IDs already in the proximity reference, or a DataFrame of novel inputs.
            predictions: Model predictions (ensemble mean), same length as `query`.
            prediction_std: Ensemble standard deviation, same length as `query`.

        Returns:
            DataFrame with columns:
                expected_residual, confidence, q_025, q_05, q_10, q_16, q_25,
                q_50, q_75, q_84, q_90, q_95, q_975
            indexed by query id (or query_id / positional index for novel queries).
        """
        if self.error_model is None:
            raise RuntimeError("UQModelV1 not fitted. Call .fit(...) first or .load(...).")

        predictions = np.asarray(predictions, dtype=float).ravel()
        prediction_std = np.asarray(prediction_std, dtype=float).ravel()

        # Auto-dispatch
        if isinstance(query, pd.DataFrame):
            feat = self.features.compute_from_query_df(query, predictions=predictions, k=self.k, training_only=False)
        else:
            ids = list(query) if not isinstance(query, list) else query
            if len(predictions) != len(ids):
                raise ValueError(f"predictions length ({len(predictions)}) must match number of queries ({len(ids)})")
            feat = self.features.compute(ids, predictions=predictions, k=self.k, training_only=False)

        X_test = self._stack_features(predictions, prediction_std, feat)
        expected_residual = self.error_model.predict(X_test)

        # Confidence: percentile rank of expected residual against cal-set distribution
        ranks = np.searchsorted(self.residual_percentiles, expected_residual, side="right") / len(
            self.residual_percentiles
        )
        confidence = np.clip(1.0 - ranks, 0.0, 1.0)

        # Build result DataFrame
        result = pd.DataFrame(
            {
                "expected_residual": expected_residual,
                "confidence": confidence,
                "q_50": predictions,
            },
            index=feat.index,
        )

        for alpha in self.confidence_levels:
            q = self.scale_factors[f"{alpha:.2f}"]
            lower = predictions - q * expected_residual
            upper = predictions + q * expected_residual
            if alpha in _QUANTILE_COLUMNS:
                lo_col, hi_col = _QUANTILE_COLUMNS[alpha]
                result[lo_col] = lower
                result[hi_col] = upper

        # Reorder columns for readability
        quantile_cols = ["q_025", "q_05", "q_10", "q_16", "q_25", "q_50", "q_75", "q_84", "q_90", "q_95", "q_975"]
        existing_q = [c for c in quantile_cols if c in result.columns]
        return result[["expected_residual", "confidence"] + existing_q]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    METADATA_FILENAME = "uq_metadata_v1.json"
    UQ_VERSION = "v1"

    def save(self, model_dir: str, save_proximity: bool = True) -> None:
        """Save fitted state to a model directory.

        Writes:
            uq_model.joblib        — pickled RandomForestRegressor
            uq_metadata_v1.json    — scale_factors, residual_percentiles, hyperparams
            uq_proximity.joblib    — pickled Proximity backend (if save_proximity=True)

        For SageMaker / self-contained inference, leave save_proximity=True so the
        full pipeline can be reconstructed from the model artifact alone. For
        workbench-internal use where the Proximity model is rebuilt on demand from
        the source FeatureSet, set save_proximity=False to keep artifacts lean.
        """
        if self.error_model is None:
            raise RuntimeError("UQModelV1 not fitted; nothing to save.")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.error_model, os.path.join(model_dir, "uq_model.joblib"))

        if save_proximity:
            joblib.dump(self._slim_proximity(self.prox), os.path.join(model_dir, "uq_proximity.joblib"))

        metadata = {
            "confidence_levels": self.confidence_levels,
            "scale_factors": self.scale_factors,
            "residual_percentiles": self.residual_percentiles.tolist(),
            "k": self.k,
            "training_only_features": self.training_only_features,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "feature_order": self.FEATURE_ORDER,
            "proximity_saved": save_proximity,
        }
        with open(os.path.join(model_dir, self.METADATA_FILENAME), "w") as fp:
            json.dump(metadata, fp, indent=2)

        log.info(f"Saved UQModelV1 to {model_dir} (proximity={'embedded' if save_proximity else 'external'})")

    @classmethod
    def load(cls, model_dir: str, prox: Optional[Proximity] = None) -> "UQModelV1":
        """Load a fitted UQModelV1 from disk.

        Args:
            model_dir: Directory containing uq_model.joblib + uq_metadata_v1.json
                (and optionally uq_proximity.joblib).
            prox: Proximity backend to use for inference-time neighbor lookups.
                If None, load the embedded proximity from uq_proximity.joblib.
                If provided, the embedded proximity is ignored — useful for
                workbench-internal use where the Proximity is freshly rebuilt from
                the latest FeatureSet on demand.

        Returns:
            A UQModelV1 ready to .predict(...).
        """
        metadata_path = os.path.join(model_dir, cls.METADATA_FILENAME)
        with open(metadata_path) as fp:
            metadata = json.load(fp)

        if prox is None:
            prox_path = os.path.join(model_dir, "uq_proximity.joblib")
            if not os.path.exists(prox_path):
                raise FileNotFoundError(
                    f"No proximity backend provided and no {prox_path} found. "
                    "Pass `prox=...` explicitly or save with save_proximity=True."
                )
            prox = joblib.load(prox_path)

        instance = cls(
            prox=prox,
            confidence_levels=metadata["confidence_levels"],
            k=metadata["k"],
            training_only_features=metadata["training_only_features"],
            n_estimators=metadata["n_estimators"],
            max_depth=metadata["max_depth"],
        )
        instance.error_model = joblib.load(os.path.join(model_dir, "uq_model.joblib"))
        instance.scale_factors = metadata["scale_factors"]
        instance.residual_percentiles = np.asarray(metadata["residual_percentiles"])
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_features(predictions: np.ndarray, prediction_std: np.ndarray, feat: pd.DataFrame) -> np.ndarray:
        """Build the (n, 5) feature matrix in canonical column order."""
        # NaN-fill: queries with no valid neighbors (rare) get neutral values
        knn_distance = np.nan_to_num(feat["knn_distance"].values, nan=0.5)
        knn_target_std = np.nan_to_num(feat["knn_target_std"].values, nan=0.0)
        local_pred_gap = np.nan_to_num(feat.get("local_pred_gap", pd.Series(0.0, index=feat.index)).values, nan=0.0)
        return np.column_stack([predictions, prediction_std, knn_distance, knn_target_std, local_pred_gap])

    @staticmethod
    def _slim_proximity(prox: Proximity) -> Proximity:
        """Return a slimmed copy of the proximity backend suitable for embedding in the
        model artifact.

        The slim is consumed exclusively by `UQModelV1.predict()` at inference time.
        The only columns the inference path reads from `prox.df` are:
            - `id_column`       — for neighbor_id values in result + `_id_to_row` cache
            - `target` column   — for aggregating into knn_target_mean / knn_target_std

        Everything else gets dropped:
            - `fingerprint` (count-FP strings, ~4 KB/row — the bulk of the artifact)
            - `smiles` (novel queries supply their own)
            - `in_model` (always True at SageMaker training time, never read at inference)
            - `prediction` / `_proba` / `residual` pass-throughs (generic Proximity
              feature for other use cases, never consumed by ResidualFeatures)

        For non-FingerprintProximity backends this is a no-op pass-through.
        """
        import copy

        slim = copy.copy(prox)  # shallow copy; we'll swap the df reference
        if not hasattr(prox, "_X_sparse") and not hasattr(prox, "fingerprint_column"):
            return prox  # Not a FingerprintProximity — leave as-is

        keep_cols = [prox.id_column]
        if prox.target and prox.target in prox.df.columns:
            keep_cols.append(prox.target)
        slim.df = prox.df[keep_cols].copy()
        return slim
