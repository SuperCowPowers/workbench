"""Cleanlab-based label quality detection for regression and classification.

Cleanlab is an **optional** dependency — not installed by ``workbench`` or any
of its extras. Users who want to use :class:`CleanlabModels` (via
:meth:`workbench.api.Model.cleanlab_model` or
:meth:`workbench.api.FeatureSet.cleanlab_model`) must install it separately::

    pip install 'cleanlab[datalab]>=2.8.0'

Cleanlab 2.8.0+ resolves the earlier Datalab/datasets 4.x incompatibility:
https://github.com/cleanlab/cleanlab/issues/1253

This module imports cleanly even when cleanlab is not installed; the
ImportError is deferred to :class:`CleanlabModels.__init__` so importing
workbench (or its api submodules) doesn't require cleanlab.
"""

import logging
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from workbench.core.artifacts.model_core import ModelType

# Optional dependency check — see module docstring.
_CLEANLAB_IMPORT_ERROR: Optional[str] = None
try:
    from cleanlab.regression.learn import CleanLearning as CleanLearningRegressor
    from cleanlab.classification import CleanLearning as CleanLearningClassifier
    from cleanlab import Datalab

    CLEANLAB_AVAILABLE = True
except ImportError as _e:
    CLEANLAB_AVAILABLE = False
    CleanLearningRegressor = None
    CleanLearningClassifier = None
    Datalab = None
    _CLEANLAB_IMPORT_ERROR = (
        f"{_e}. "
        "cleanlab is an optional workbench dependency — install with: "
        "pip install 'cleanlab[datalab]>=2.8.0'"
    )

# Regressor types for convenience
REGRESSOR_TYPES = [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]

# Set up logging
log = logging.getLogger("workbench")


class CleanlabModels:
    """Cleanlab label-quality and data-quality analysis with shared data prep.

    Prepares the data once, then provides access to the underlying cleanlab
    objects along with workbench helpers that join results back to the ID column.

    The cleanlab objects are exposed directly and unmodified — use them with
    cleanlab's own API (see https://docs.cleanlab.ai):

        - ``clean_learning()`` → cleanlab ``CleanLearning`` (fitted)
        - ``datalab()`` → cleanlab ``Datalab`` (with ``find_issues`` already run)

    The workbench helpers below are the recommended surface for most uses; they
    return DataFrames keyed by the ID column (or a scalar, for aleatoric):

        - ``label_issues()`` → DataFrame, worst label quality first
        - ``epistemic_uncertainty()`` → DataFrame, highest model uncertainty first (regression)
        - ``aleatoric_uncertainty()`` → float, dataset-level irreducible noise (regression)

    Example:
        ```python
        cleanlab = CleanlabModels(df, "id", features, "target", ModelType.REGRESSOR)

        # Workbench helpers (ID-joined results)
        issues = cleanlab.label_issues()
        uncertainty = cleanlab.epistemic_uncertainty()

        # Or work with the native cleanlab objects directly
        cleanlab.datalab().report()
        ```
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: str,
        model_type: ModelType = ModelType.REGRESSOR,
    ):
        """Initialize CleanlabModels with data preparation.

        Args:
            df: DataFrame containing data for analysis.
            id_column: Name of the column used as the identifier.
            features: List of feature column names.
            target: Name of the target column.
            model_type: ModelType (REGRESSOR, CLASSIFIER, etc.).
        """
        if not CLEANLAB_AVAILABLE:
            raise ImportError(_CLEANLAB_IMPORT_ERROR)

        self.id_column = id_column
        self.target = target
        self.model_type = model_type

        # Filter to numeric features only
        numeric_cols = df.select_dtypes(include=["number"]).columns
        non_numeric = [f for f in features if f not in numeric_cols]
        if non_numeric:
            log.warning(f"Excluding non-numeric features: {non_numeric}")
            features = [f for f in features if f in numeric_cols]
        self.features = features

        # Prepare clean data (shared by both models)
        self._clean_df = df.dropna(subset=features + [target])[[id_column] + features + [target]].copy()
        self._clean_df = self._clean_df.reset_index(drop=True)
        self._X = self._clean_df[features].values
        self._y = self._clean_df[target].values

        # For classification, encode labels
        self._label_encoder: Optional[LabelEncoder] = None
        self._y_encoded = self._y
        if model_type == ModelType.CLASSIFIER:
            self._label_encoder = LabelEncoder()
            self._y_encoded = self._label_encoder.fit_transform(self._y)

        # Lazy-loaded models
        self._clean_learning = None
        self._datalab = None

    def clean_learning(self):
        """Get the native cleanlab CleanLearning model (fitted, lazily built).

        The returned object is the unmodified cleanlab model — use it with
        cleanlab's own API. For ID-joined results, prefer the workbench helpers
        (``label_issues()``, ``epistemic_uncertainty()``, ``aleatoric_uncertainty()``).

        Returns:
            CleanLearning: Fitted cleanlab model.
        """
        if self._clean_learning is not None:
            return self._clean_learning

        if self.model_type == ModelType.CLASSIFIER:
            log.info("Building CleanLearning model (classification)...")
            cl_model = CleanLearningClassifier(
                HistGradientBoostingClassifier(),
                find_label_issues_kwargs={"n_jobs": 1},
            )
            cl_model.fit(self._X, self._y_encoded)
        else:
            log.info("Building CleanLearning model (regression)...")
            cl_model = CleanLearningRegressor(HistGradientBoostingRegressor())
            cl_model.fit(self._X, self._y)

        self._clean_learning = cl_model
        return cl_model

    def datalab(self):
        """Get the native cleanlab Datalab instance (with find_issues already run).

        The returned object is the unmodified cleanlab Datalab — use it with
        cleanlab's own API (e.g. ``report()``, ``get_issues()``, ``get_issue_summary()``).

        Note: For classification, this builds the CleanLearning model first (if
        not already built) to reuse its classifier for pred_probs.

        Returns:
            Datalab: Cleanlab Datalab instance.
        """
        if self._datalab is not None:
            return self._datalab

        log.info("Building Datalab model...")

        # Create DataFrame with only numeric columns (features + target) for Datalab
        datalab_df = self._clean_df[self.features + [self.target]]

        # Create Datalab instance
        if self.model_type == ModelType.CLASSIFIER:
            lab = Datalab(data=datalab_df, label_name=self.target)
            # Build CleanLearning first to reuse its classifier for pred_probs
            cl = self.clean_learning()
            pred_probs = cl.clf.predict_proba(self._X)
            lab.find_issues(features=self._X, pred_probs=pred_probs)
        else:
            lab = Datalab(data=datalab_df, label_name=self.target, task="regression")
            lab.find_issues(features=self._X)

        self._datalab = lab
        return lab

    def label_issues(self) -> pd.DataFrame:
        """Detected label issues, keyed by ID column and worst quality first.

        Returns:
            pd.DataFrame: One row per sample with the ID column inserted first,
                sorted ascending by ``label_quality``. For classification,
                ``given_label`` and ``predicted_label`` are decoded back to the
                original labels.
        """
        cl = self.clean_learning()
        issues = cl.get_label_issues().copy()
        issues.insert(0, self.id_column, self._clean_df[self.id_column].values)

        if self.model_type == ModelType.CLASSIFIER and self._label_encoder is not None:
            for col in ["given_label", "predicted_label"]:
                if col in issues.columns:
                    issues[col] = self._label_encoder.inverse_transform(issues[col])

        n_issues = int(issues["is_label_issue"].sum())
        log.info(f"CleanLearning: {n_issues} potential label issues out of {len(self._clean_df)} samples")
        return issues.sort_values("label_quality").reset_index(drop=True)

    def epistemic_uncertainty(self) -> pd.DataFrame:
        """Per-sample epistemic (model) uncertainty, keyed by ID column.

        Epistemic uncertainty is the reducible component — high values flag
        samples the model is unsure about. Regression models only.

        Returns:
            pd.DataFrame: One row per sample with the ID column and an
                ``epistemic_uncertainty`` column, sorted descending.
        """
        self._require_regression("epistemic_uncertainty")
        cl = self.clean_learning()
        values = cl.get_epistemic_uncertainty(self._X, self._y)
        return (
            pd.DataFrame(
                {
                    self.id_column: self._clean_df[self.id_column].values,
                    "epistemic_uncertainty": values,
                }
            )
            .sort_values("epistemic_uncertainty", ascending=False)
            .reset_index(drop=True)
        )

    def aleatoric_uncertainty(self) -> float:
        """Dataset-level aleatoric (irreducible) noise estimate.

        Aleatoric uncertainty is the irreducible component — inherent noise in
        the data that more modeling can't remove. Regression models only.

        Returns:
            float: Single dataset-level aleatoric uncertainty estimate.
        """
        self._require_regression("aleatoric_uncertainty")
        cl = self.clean_learning()
        residual = cl.predict(self._X) - self._y
        return cl.get_aleatoric_uncertainty(self._X, residual)

    def _require_regression(self, method_name: str):
        """Raise a clear error if a regression-only method is called on a classifier."""
        if self.model_type == ModelType.CLASSIFIER:
            raise TypeError(f"{method_name}() is only available for regression models")


if __name__ == "__main__":
    from workbench.api import FeatureSet, Model
    import numpy as np

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create a sample DataFrame with some noisy points
    np.random.seed(42)
    n_samples = 100

    # Generate clean data: y = 2*x1 + 3*x2 + noise
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    y_clean = 2 * x1 + 3 * x2 + np.random.randn(n_samples) * 0.1

    # Add some noisy points (last 10 samples)
    y_noisy = y_clean.copy()
    y_noisy[-10:] += np.random.randn(10) * 20  # Large noise

    data = {
        "ID": [f"sample_{i}" for i in range(n_samples)],
        "Feature1": x1,
        "Feature2": x2,
        "target": y_noisy,
    }
    df = pd.DataFrame(data)

    print("=" * 80)
    print("Testing CleanlabModels with synthetic data...")
    print("=" * 80)

    cleanlab_models = CleanlabModels(
        df,
        id_column="ID",
        features=["Feature1", "Feature2"],
        target="target",
    )

    label_issues = cleanlab_models.label_issues()
    print("\nLabel issues (worst first, with ID column):")
    print(label_issues.head(10))

    # Check if our artificially noisy samples are detected
    noisy_ids = [f"sample_{i}" for i in range(90, 100)]
    worst_10 = label_issues.head(10)
    detected = worst_10[worst_10["ID"].isin(noisy_ids)]
    print(f"\nOf 10 noisy samples, {len(detected)} appear in worst 10")

    # Native Datalab object
    print("\n" + "=" * 80)
    print("Testing Datalab...")
    print("=" * 80)
    lab = cleanlab_models.datalab()
    print(f"Datalab type: {type(lab)}")
    print(f"Datalab issues shape: {lab.get_issues().shape}")
    lab.report(num_examples=3)

    # Test with real AQSol regression data
    print("\n" + "=" * 80)
    print("Testing with AQSol regression data...")
    print("=" * 80)
    fs = FeatureSet("aqsol_features")
    df = fs.pull_dataframe()
    model = Model("aqsol-regression")
    features = model.features()
    target = model.target()

    cleanlab_models = CleanlabModels(
        df,
        id_column=fs.id_column,
        features=features,
        target=target,
    )

    label_issues = cleanlab_models.label_issues()
    print("\nLabel issues summary:")
    print(f"Total samples: {len(label_issues)}")
    print(f"Flagged as issues: {label_issues['is_label_issue'].sum()}")

    print("\nWorst label quality samples:")
    print(label_issues.head(10))

    print("\nLabel quality distribution:")
    print(label_issues["label_quality"].describe())

    # Uncertainty estimates (regression only)
    print("\nTesting uncertainty estimates...")
    aleatoric = cleanlab_models.aleatoric_uncertainty()
    print(f"Aleatoric: Data noise (irreducible) = {aleatoric}")
    epistemic = cleanlab_models.epistemic_uncertainty()
    print("Epistemic: Model uncertainty (reducible), highest first:")
    print(epistemic.head(10))

    # Native Datalab report
    print("\n" + "=" * 80)
    print("Testing Datalab report (regression)...")
    print("=" * 80)
    lab = cleanlab_models.datalab()
    lab.report(num_examples=3)
