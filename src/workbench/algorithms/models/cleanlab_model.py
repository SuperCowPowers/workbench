"""Cleanlab-based label quality detection for regression and classification.

This module provides a factory function to create a fitted cleanlab CleanLearning
model for regression or classification.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List
import logging

# Cleanlab imports
from cleanlab.regression.learn import CleanLearning as CleanLearningRegressor
from cleanlab.classification import CleanLearning as CleanLearningClassifier

# Workbench imports
from workbench.core.artifacts.model_core import ModelType

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
        model_type: ModelType.REGRESSOR or ModelType.CLASSIFIER.

    Returns:
        CleanLearning: A fitted cleanlab model. Use get_label_issues() to get
        a DataFrame sorted by label_quality (worst first) with id column included.

    Example:
        ```python
        from workbench.algorithms.models.cleanlab_model import create_cleanlab_model

        cl_model = create_cleanlab_model(df, id_column="id", features=feature_list, target="target")
        label_issues = cl_model.get_label_issues()

        # Already sorted by label_quality (worst first)
        worst = label_issues.head(20)
        ```

    References:
        cleanlab: https://github.com/cleanlab/cleanlab
        Documentation: https://docs.cleanlab.ai/stable/tutorials/regression.html
    """
    # Filter out non-numeric features
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric = [f for f in features if f not in numeric_cols]
    if non_numeric:
        log.warning(f"Non-numeric features {non_numeric} aren't currently supported, excluding them")
    features = [f for f in features if f not in non_numeric]

    # Drop NaN rows in features and target, keep id column
    clean_df = df.dropna(subset=features + [target])[[id_column] + features + [target]].copy()
    clean_df = clean_df.reset_index(drop=True)

    X = clean_df[features].values
    y = clean_df[target].values

    # Create model based on model_type
    if model_type == ModelType.CLASSIFIER:
        log.info("Building CleanLearning model (classification)...")
        # Encode string labels to integers for cleanlab
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # Disable multiprocessing (n_jobs=1) to prevent worker processes from
        # re-importing workbench modules (config, AWS credentials, license checks)
        cl_model = CleanLearningClassifier(
            HistGradientBoostingClassifier(),
            find_label_issues_kwargs={"n_jobs": 1},
        )
        log.info("  Finding label issues via cross-validation...")
        cl_model.fit(X, y_encoded)
    else:
        log.info("Building CleanLearning model (regression)...")
        cl_model = CleanLearningRegressor(HistGradientBoostingRegressor())
        log.info("  Finding label issues via cross-validation...")
        cl_model.fit(X, y)

    # Monkey-patch get_label_issues to include id column, sort, and reset index
    original_get_label_issues = cl_model.get_label_issues

    def get_label_issues_with_id():
        label_issues = original_get_label_issues().copy()
        label_issues.insert(0, id_column, clean_df[id_column].values)
        # For classification, decode labels back to original strings
        if model_type == ModelType.CLASSIFIER:
            if "given_label" in label_issues.columns:
                label_issues["given_label"] = label_encoder.inverse_transform(label_issues["given_label"])
            if "predicted_label" in label_issues.columns:
                label_issues["predicted_label"] = label_encoder.inverse_transform(label_issues["predicted_label"])
        return label_issues.sort_values("label_quality").reset_index(drop=True)

    cl_model.get_label_issues = get_label_issues_with_id

    n_issues = original_get_label_issues()["is_label_issue"].sum()
    log.info(f"  Found {n_issues} potential label issues out of {len(clean_df)} samples")
    log.info("CleanLearning model built successfully")

    return cl_model


# Testing
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


    print("\nLabel quality distribution:")
    print(label_issues["label_quality"].describe())
