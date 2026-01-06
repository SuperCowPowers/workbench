"""Cleanlab-based label quality detection for regression and classification.

Note: Users must install cleanlab separately: pip install cleanlab
"""

import logging
from typing import List, Optional

import datasets
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from workbench.core.artifacts.model_core import ModelType

# Check datasets version - Datalab has a bug with datasets>=4.0.0
# See: https://github.com/cleanlab/cleanlab/issues/1253
_datasets_major = int(datasets.__version__.split(".")[0])
if _datasets_major >= 4:
    raise ImportError(
        "cleanlab's Datalab requires datasets<4.0.0 due to a known bug.\n"
        "See: https://github.com/cleanlab/cleanlab/issues/1253\n"
        "Fix: pip install 'datasets<4.0.0'"
    )

# Check for cleanlab package
try:
    from cleanlab.regression.learn import CleanLearning as CleanLearningRegressor
    from cleanlab.classification import CleanLearning as CleanLearningClassifier
    from cleanlab import Datalab

    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    CleanLearningRegressor = None
    CleanLearningClassifier = None
    Datalab = None

# Regressor types for convenience
REGRESSOR_TYPES = [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]

# Set up logging
log = logging.getLogger("workbench")


class CleanlabModels:
    """Factory class for cleanlab models with shared data preparation.

    This class handles data preparation once and provides lazy-loaded access
    to both CleanLearning and Datalab models. Each model is only created
    when first requested, and the prepared data is shared between them.

    Attributes:
        id_column: Name of the ID column in the data.
        features: List of feature column names.
        target: Name of the target column.
        model_type: ModelType (REGRESSOR, CLASSIFIER, etc.).

    Example:
        ```python
        cleanlab = CleanlabModels(df, "id", features, "target", ModelType.REGRESSOR)

        # Get CleanLearning model for label issues and uncertainty
        cl = cleanlab.clean_learning()
        issues = cl.get_label_issues()

        # Get Datalab for comprehensive data quality report
        lab = cleanlab.datalab()
        lab.report()
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
            raise ImportError("cleanlab is not installed. Install with: pip install 'cleanlab[datalab]'")

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
        """Get the CleanLearning model (fitted, with label issues computed).

        Returns the cleanlab CleanLearning model with enhanced get_label_issues()
        that includes the ID column, sorts by label quality, and decodes labels.

        Returns:
            CleanLearning: Fitted cleanlab model with methods like:
                - get_label_issues(): DataFrame with id_column, sorted by label_quality
                - predict(X): Make predictions
                - For regression: get_epistemic_uncertainty(), get_aleatoric_uncertainty()
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

        # Enhance get_label_issues to include id column, sort, and decode labels
        original_get_label_issues = cl_model.get_label_issues
        id_column = self.id_column
        clean_df = self._clean_df
        model_type = self.model_type
        label_encoder = self._label_encoder

        def get_label_issues_enhanced():
            issues = original_get_label_issues().copy()
            issues.insert(0, id_column, clean_df[id_column].values)
            if model_type == ModelType.CLASSIFIER and label_encoder is not None:
                for col in ["given_label", "predicted_label"]:
                    if col in issues.columns:
                        issues[col] = label_encoder.inverse_transform(issues[col])
            return issues.sort_values("label_quality").reset_index(drop=True)

        cl_model.get_label_issues = get_label_issues_enhanced

        # For regression, enhance uncertainty methods to use stored data and return DataFrames
        if model_type != ModelType.CLASSIFIER:
            X = self._X
            y = self._y
            original_get_aleatoric = cl_model.get_aleatoric_uncertainty
            original_get_epistemic = cl_model.get_epistemic_uncertainty

            def get_aleatoric_uncertainty_enhanced():
                residual = cl_model.predict(X) - y
                return original_get_aleatoric(X, residual)

            def get_epistemic_uncertainty_enhanced():
                values = original_get_epistemic(X, y)
                return (
                    pd.DataFrame(
                        {
                            id_column: clean_df[id_column].values,
                            "epistemic_uncertainty": values,
                        }
                    )
                    .sort_values("epistemic_uncertainty", ascending=False)
                    .reset_index(drop=True)
                )

            cl_model.get_aleatoric_uncertainty = get_aleatoric_uncertainty_enhanced
            cl_model.get_epistemic_uncertainty = get_epistemic_uncertainty_enhanced

        n_issues = original_get_label_issues()["is_label_issue"].sum()
        log.info(f"CleanLearning: {n_issues} potential label issues out of {len(self._clean_df)} samples")

        self._clean_learning = cl_model
        return cl_model

    def datalab(self):
        """Get the Datalab instance (with find_issues already called).

        Returns the native cleanlab Datalab for comprehensive data quality
        analysis. Issues have already been detected.

        Note: For classification, this will build the CleanLearning model first
        (if not already built) to reuse its classifier for pred_probs.

        Returns:
            Datalab: Cleanlab Datalab instance with methods like:
                - report(): Print comprehensive data quality report
                - get_issues(): DataFrame with all detected issues
                - get_issue_summary(): Summary statistics
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


# Keep the old function for backwards compatibility
def create_cleanlab_model(
    df: pd.DataFrame,
    id_column: str,
    features: List[str],
    target: str,
    model_type: ModelType = ModelType.REGRESSOR,
):
    """Create a CleanlabModels instance for label quality detection.

    Args:
        df: DataFrame containing data for label quality detection.
        id_column: Name of the column used as the identifier.
        features: List of feature column names.
        target: Name of the target column.
        model_type: ModelType (REGRESSOR, CLASSIFIER, etc.).

    Returns:
        CleanlabModels: Factory providing access to CleanLearning and Datalab models.

    Example:
        ```python
        cleanlab = create_cleanlab_model(df, "id", features, "target")

        # Get CleanLearning model and label issues
        cl = cleanlab.clean_learning()
        issues = cl.get_label_issues()  # Includes ID column, sorted by quality

        # Get Datalab for comprehensive data quality report
        lab = cleanlab.datalab()
        lab.report()
        ```

    References:
        cleanlab: https://github.com/cleanlab/cleanlab
    """
    return CleanlabModels(df, id_column, features, target, model_type)


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

    # Create CleanlabModels instance
    cleanlab_models = create_cleanlab_model(
        df,
        id_column="ID",
        features=["Feature1", "Feature2"],
        target="target",
    )

    # Get CleanLearning model and test get_label_issues
    cl = cleanlab_models.clean_learning()
    print(f"CleanLearning type: {type(cl)}")

    label_issues = cl.get_label_issues()
    print("\nLabel issues (worst first, with ID column):")
    print(label_issues.head(10))

    # Check if our artificially noisy samples are detected
    noisy_ids = [f"sample_{i}" for i in range(90, 100)]
    worst_10 = label_issues.head(10)
    detected = worst_10[worst_10["ID"].isin(noisy_ids)]
    print(f"\nOf 10 noisy samples, {len(detected)} appear in worst 10")

    # Test Datalab
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

    cleanlab_models = create_cleanlab_model(
        df,
        id_column=fs.id_column,
        features=features,
        target=target,
    )

    # Get CleanLearning and label issues
    cl = cleanlab_models.clean_learning()
    label_issues = cl.get_label_issues()
    print("\nLabel issues summary:")
    print(f"Total samples: {len(label_issues)}")
    print(f"Flagged as issues: {label_issues['is_label_issue'].sum()}")

    print("\nWorst label quality samples:")
    print(label_issues.head(10))

    print("\nLabel quality distribution:")
    print(label_issues["label_quality"].describe())

    # Test uncertainty estimates (regression only)
    print("\nTesting uncertainty estimates...")
    aleatoric = cl.get_aleatoric_uncertainty(cleanlab_models._X, cl.predict(cleanlab_models._X) - cleanlab_models._y)
    print(f"Aleatoric: Data noise (irreducible) = {aleatoric}")
    epistemic = cl.get_epistemic_uncertainty(cleanlab_models._X, cleanlab_models._y)
    print(f"Epistemic: Model uncertainty (reducible) = {epistemic[:10]} ...")

    # Test Datalab report
    print("\n" + "=" * 80)
    print("Testing Datalab report (regression)...")
    print("=" * 80)
    lab = cleanlab_models.datalab()
    lab.report(num_examples=3)
