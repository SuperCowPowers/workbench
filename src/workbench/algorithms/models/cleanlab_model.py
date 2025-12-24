"""Cleanlab-based label quality detection for regression and classification.

Note: Users must install cleanlab separately: pip install cleanlab
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List
import logging

# Check for cleanlab package
try:
    from cleanlab.regression.learn import CleanLearning as CleanLearningRegressor
    from cleanlab.classification import CleanLearning as CleanLearningClassifier

    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    CleanLearningRegressor = None
    CleanLearningClassifier = None

# Workbench imports
from workbench.core.artifacts.model_core import ModelType

# Regressor types for convenience
REGRESSOR_TYPES = [ModelType.REGRESSOR, ModelType.UQ_REGRESSOR, ModelType.ENSEMBLE_REGRESSOR]

# Set up logging
log = logging.getLogger("workbench")


def create_cleanlab_model(
    df: pd.DataFrame,
    id_column: str,
    features: List[str],
    target: str,
    model_type: ModelType = ModelType.REGRESSOR,
):
    """Create a fitted CleanLearning model for label quality detection.

    Args:
        df: DataFrame containing data for label quality detection.
        id_column: Name of the column used as the identifier.
        features: List of feature column names.
        target: Name of the target column.
        model_type: ModelType (REGRESSOR, CLASSIFIER, etc.).

    Returns:
        CleanLearning: A fitted cleanlab model with enhanced get_label_issues().

    Example:
        ```python
        cl_model = model.cleanlab_model()
        label_issues = cl_model.get_label_issues()  # Sorted by label_quality (worst first)
        ```

    References:
        cleanlab: https://github.com/cleanlab/cleanlab
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError("cleanlab is not installed. Install with: pip install cleanlab")

    # Filter to numeric features only
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric = [f for f in features if f not in numeric_cols]
    if non_numeric:
        log.warning(f"Excluding non-numeric features: {non_numeric}")
        features = [f for f in features if f in numeric_cols]

    # Prepare clean data
    clean_df = df.dropna(subset=features + [target])[[id_column] + features + [target]].copy()
    clean_df = clean_df.reset_index(drop=True)
    X = clean_df[features].values
    y = clean_df[target].values

    # Build model based on type
    if model_type == ModelType.CLASSIFIER:
        log.info("Building CleanLearning model (classification)...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # n_jobs=1 prevents worker processes from re-importing workbench modules
        cl_model = CleanLearningClassifier(
            HistGradientBoostingClassifier(),
            find_label_issues_kwargs={"n_jobs": 1},
        )
        cl_model.fit(X, y_encoded)
    else:
        log.info("Building CleanLearning model (regression)...")
        cl_model = CleanLearningRegressor(HistGradientBoostingRegressor())
        cl_model.fit(X, y)

    # Enhance get_label_issues to include id column, sort, and decode labels
    original_get_label_issues = cl_model.get_label_issues

    def get_label_issues_with_id():
        issues = original_get_label_issues().copy()
        issues.insert(0, id_column, clean_df[id_column].values)
        if model_type == ModelType.CLASSIFIER:
            for col in ["given_label", "predicted_label"]:
                if col in issues.columns:
                    issues[col] = label_encoder.inverse_transform(issues[col])
        return issues.sort_values("label_quality").reset_index(drop=True)

    cl_model.get_label_issues = get_label_issues_with_id

    # For regression, add no-arg wrappers for uncertainty methods
    if model_type in REGRESSOR_TYPES:
        orig_epistemic = cl_model.get_epistemic_uncertainty
        orig_aleatoric = cl_model.get_aleatoric_uncertainty

        def get_epistemic_uncertainty():
            """Get epistemic uncertainty (model uncertainty) as DataFrame sorted by uncertainty (worst first)."""
            uncertainty = orig_epistemic(X, y)
            result = pd.DataFrame({id_column: clean_df[id_column].values, "epistemic_uncertainty": uncertainty})
            return result.sort_values("epistemic_uncertainty", ascending=False).reset_index(drop=True)

        def get_aleatoric_uncertainty():
            """Get aleatoric uncertainty (data noise) estimate."""
            return orig_aleatoric(X, cl_model.predict(X) - y)

        cl_model.get_epistemic_uncertainty = get_epistemic_uncertainty
        cl_model.get_aleatoric_uncertainty = get_aleatoric_uncertainty

    n_issues = original_get_label_issues()["is_label_issue"].sum()
    log.info(f"CleanLearning: {n_issues} potential label issues out of {len(clean_df)} samples")

    return cl_model


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
    print("Testing create_cleanlab_model...")
    print("=" * 80)

    # Create cleanlab model
    cl_model = create_cleanlab_model(
        df,
        id_column="ID",
        features=["Feature1", "Feature2"],
        target="target",
    )

    # Get label issues - already sorted by label_quality (worst first)
    label_issues = cl_model.get_label_issues()
    print("\nLabel issues (worst first, with ID column):")
    print(label_issues.head(10))

    # Check if our artificially noisy samples are detected
    noisy_ids = [f"sample_{i}" for i in range(90, 100)]
    worst_10 = label_issues.head(10)
    detected = worst_10[worst_10["ID"].isin(noisy_ids)]
    print(f"\nOf 10 noisy samples, {len(detected)} appear in worst 10")

    # Test a classification example with real data
    fs = FeatureSet("aqsol_features")
    df = fs.pull_dataframe()
    print("\n" + "=" * 80)
    print("Testing classification example...")
    print("=" * 80)
    model = Model("aqsol-class")
    features = model.features()
    target = model.target()
    cl_model = create_cleanlab_model(
        df,
        id_column=fs.id_column,
        features=features,
        target=target,
        model_type=ModelType.CLASSIFIER,
    )

    label_issues = cl_model.get_label_issues()
    print("\nClassification label issues summary:")
    print(f"Total samples: {len(label_issues)}")
    print(f"Flagged as issues: {label_issues['is_label_issue'].sum()}")

    print("\nWorst label quality samples:")
    print(label_issues.head(10))

    # Test a regression example with real data
    print("\n" + "=" * 80)
    print("Testing with AQSol data...")
    print("=" * 80)
    model = Model("aqsol-regression")
    features = model.features()
    target = model.target()

    cl_model = create_cleanlab_model(
        df,
        id_column=fs.id_column,
        features=features,
        target=target,
    )

    label_issues = cl_model.get_label_issues()
    print("\nLabel issues summary:")
    print(f"Total samples: {len(label_issues)}")
    print(f"Flagged as issues: {label_issues['is_label_issue'].sum()}")

    print("\nWorst label quality samples:")
    print(label_issues.head(10))

    print("\nLabel quality distribution:")
    print(label_issues["label_quality"].describe())

    print("\nTesting uncertainty estimates...")
    aleatoric = cl_model.get_aleatoric_uncertainty()
    print(f"Aleatoric: Data noise (irreducible) = {aleatoric}")
    epistemic = cl_model.get_epistemic_uncertainty()
    print(f"Epistemic: Model uncertainty (reducible) = {epistemic[:10]} ...")
