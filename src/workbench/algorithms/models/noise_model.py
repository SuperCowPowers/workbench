import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from typing import List
import logging

from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

# Set up logging
log = logging.getLogger("workbench")


class NoiseModel:
    """Composite noise detection for regression data using multiple complementary signals.

    The NoiseModel identifies potentially noisy or problematic samples in regression datasets
    by combining three independent signals:

    1. **Underfit Model Residuals**: A deliberately simple XGBoost model (low depth, few trees)
       that captures only the main trends. High residuals indicate samples in complex regions
       or unusual areas of the feature space.

    2. **Overfit Model Residuals**: A deliberately complex XGBoost model (deep trees, many
       iterations, no regularization) that attempts to memorize the training data. High residuals
       here indicate samples the model *cannot* fit even when trying to memorize - a strong
       signal of label noise. This is the "training error" approach validated in:
       "Denoising Drug Discovery Data for Improved ADMET Property Prediction" (Merck, JCIM 2024)

    3. **High Target Gradient (HTG)**: Using the Proximity class, measures disagreement between
       a sample's target value and its neighbors in feature space. High gradients indicate
       activity cliffs or potential measurement errors where similar compounds have very
       different target values.

    The combined noise score weights the overfit residual signal more heavily (2x) based on
    the paper's finding that training error is the most reliable noise detector for regression.

    Example:
        ```python
        from workbench.algorithms.models.noise_model import NoiseModel

        # Create noise model
        noise_model = NoiseModel(df, id_column="id", features=feature_list, target="target")

        # Get noise scores for all samples
        scores_df = noise_model.get_scores()

        # Get sample weights for training (lower weight for noisy samples)
        weights = noise_model.get_sample_weights(strategy="inverse")

        # Get clean subset (bottom 90% by noise score)
        clean_df = noise_model.get_clean_subset(percentile=90)

        # Find samples with same features but different targets (definite noise)
        conflicts = noise_model.coincident_conflicts()
        ```

    References:
        Adrian, M., Chung, Y., & Cheng, A. C. (2024). Denoising Drug Discovery Data for
        Improved ADMET Property Prediction. J. Chem. Inf. Model., 64(16), 6324-6337.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        features: List[str],
        target: str,
    ):
        """
        Initialize the NoiseModel class.

        Args:
            df: DataFrame containing data for noise detection.
            id_column: Name of the column used as the identifier.
            features: List of feature column names.
            target: Name of the target column.
        """
        self.id_column = id_column
        self.target = target

        # Filter out non-numeric features
        self.features = self._validate_features(df, features)

        # Drop NaN rows in features and target
        self.df = df.dropna(subset=self.features + [self.target]).copy()

        # Compute target stats for normalization
        self.target_std = self.df[self.target].std()
        self.target_range = self.df[self.target].max() - self.df[self.target].min()

        # Build all component models
        self._build_models()

        # Precompute all noise signals
        self._precompute_signals()

    def get_scores(self) -> pd.DataFrame:
        """
        Get noise scores for all samples.

        Returns:
            DataFrame with id, individual signal columns, and combined noise_score
        """
        result = self.df[[self.id_column, self.target]].copy()
        result["underfit_residual"] = self.df["underfit_residual"]
        result["overfit_residual"] = self.df["overfit_residual"]
        result["htg_score"] = self.df["htg_score"]
        result["noise_score"] = self.df["noise_score"]
        return result.sort_values("noise_score", ascending=False).reset_index(drop=True)

    def get_sample_weights(self, strategy: str = "inverse") -> pd.Series:
        """
        Get sample weights for training, indexed by id_column.

        Args:
            strategy: Weighting strategy
                - "inverse": 1 / (1 + noise_score)
                - "soft": 1 - noise_score (clipped to [0.1, 1.0])
                - "threshold": 1.0 if noise_score < median, else 0.5

        Returns:
            Series of weights indexed by id_column
        """
        scores = self.df.set_index(self.id_column)["noise_score"]

        if strategy == "inverse":
            weights = 1.0 / (1.0 + scores)
        elif strategy == "soft":
            weights = (1.0 - scores).clip(lower=0.1, upper=1.0)
        elif strategy == "threshold":
            median_score = scores.median()
            weights = (scores < median_score).apply(lambda x: 1.0 if x else 0.5)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return weights

    def get_clean_subset(self, percentile: float = 90.0) -> pd.DataFrame:
        """
        Get a subset of data with lowest noise scores.

        Args:
            percentile: Keep samples below this percentile of noise score (default: 90 = bottom 90%)

        Returns:
            DataFrame of "clean" samples
        """
        threshold = np.percentile(self.df["noise_score"], percentile)
        return self.df[self.df["noise_score"] <= threshold].copy()

    def get_noisy_samples(self, top_percent: float = 10.0) -> pd.DataFrame:
        """
        Get samples with highest noise scores.

        Args:
            top_percent: Percentage of noisiest samples to return (default: 10%)

        Returns:
            DataFrame of noisy samples, sorted by noise_score descending
        """
        percentile = 100 - top_percent
        threshold = np.percentile(self.df["noise_score"], percentile)
        noisy = self.df[self.df["noise_score"] >= threshold].copy()
        return noisy.sort_values("noise_score", ascending=False).reset_index(drop=True)

    def coincident_conflicts(self, distance_threshold: float = 1e-5) -> pd.DataFrame:
        """
        Find samples that map to the same point in feature space but have different targets.

        These are definitive noise - same features, different target values.

        Args:
            distance_threshold: Maximum distance to consider "coincident" (default: 1e-5)

        Returns:
            DataFrame of coincident conflicts with their target differences
        """
        # Use proximity to find coincident points
        coincident = self.df[self.df["nn_distance"] < distance_threshold].copy()

        if len(coincident) == 0:
            return pd.DataFrame(columns=[self.id_column, self.target, "nn_id", "nn_target", "nn_target_diff"])

        return (
            coincident[[self.id_column, self.target, "nn_id", "nn_target", "nn_target_diff", "noise_score"]]
            .sort_values("nn_target_diff", ascending=False)
            .reset_index(drop=True)
        )

    def _validate_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove non-numeric features and log warnings."""
        non_numeric = [f for f in features if f not in df.select_dtypes(include=["number"]).columns]
        if non_numeric:
            log.warning(f"Non-numeric features {non_numeric} aren't currently supported, excluding them")
        return [f for f in features if f not in non_numeric]

    def _build_models(self) -> None:
        """Build the underfit, overfit, and proximity models."""
        log.info("Building noise detection models...")

        X = self.df[self.features]
        y = self.df[self.target]

        # Underfit model: intentionally simple (high bias)
        log.info("  Fitting underfit model...")
        self.underfit_model = XGBRegressor(
            max_depth=2,
            n_estimators=20,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        self.underfit_model.fit(X, y)

        # Overfit model: intentionally complex (high variance, low regularization)
        log.info("  Fitting overfit model...")
        self.overfit_model = XGBRegressor(
            max_depth=12,
            n_estimators=500,
            learning_rate=0.1,
            reg_lambda=0.0,
            reg_alpha=0.0,
            min_child_weight=1,
            random_state=42,
            verbosity=0,
        )
        self.overfit_model.fit(X, y)

        # Proximity model for feature space analysis
        log.info("  Building proximity model...")
        self.proximity = FeatureSpaceProximity(
            self.df,
            id_column=self.id_column,
            features=self.features,
            target=self.target,
        )

        # Copy proximity metrics to our df
        self.df["nn_distance"] = self.proximity.df["nn_distance"].values
        self.df["nn_id"] = self.proximity.df["nn_id"].values
        self.df["nn_target"] = self.proximity.df["nn_target"].values
        self.df["nn_target_diff"] = self.proximity.df["nn_target_diff"].values

        log.info("Noise detection models built successfully")

    def _precompute_signals(self) -> None:
        """Precompute all noise signals for every sample."""
        log.info("Precomputing noise signals...")

        X = self.df[self.features]
        y = self.df[self.target].values

        # Underfit residuals (normalized by target std)
        underfit_pred = self.underfit_model.predict(X)
        self.df["underfit_residual"] = np.abs(y - underfit_pred) / self.target_std

        # Overfit residuals (normalized by target std)
        # This is the key "training error" signal from the paper
        overfit_pred = self.overfit_model.predict(X)
        self.df["overfit_residual"] = np.abs(y - overfit_pred) / self.target_std

        # HTG score: neighbor disagreement (normalized by target std)
        # Using nn_target_diff directly, normalized
        self.df["htg_score"] = self.df["nn_target_diff"] / self.target_std

        # Combine into overall noise score
        # Scale each component to [0, 1] using percentile ranks, then average
        self.df["noise_score"] = self._compute_combined_score()

        log.info("Noise signals precomputed successfully")

    def _compute_combined_score(self) -> np.ndarray:
        """
        Combine individual signals into a single noise score.

        Uses percentile ranks to normalize each signal to [0, 1], then averages.
        Overfit residual gets higher weight as it's the most validated signal (per the paper).
        """
        # Convert to percentile ranks (0-1 scale)
        overfit_rank = self.df["overfit_residual"].rank(pct=True)
        htg_rank = self.df["htg_score"].rank(pct=True)

        # Weighted average: overfit gets 2x weight based on paper's findings
        # that training error is the best noise detector
        combined = (2.0 * overfit_rank + 1.0 * htg_rank) / 3.0

        return combined.values


# Testing the NoiseModel class
if __name__ == "__main__":

    from workbench.api import FeatureSet, Model

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
    y_noisy[-10:] += np.random.randn(10) * 5  # Large noise

    data = {
        "ID": [f"sample_{i}" for i in range(n_samples)],
        "Feature1": x1,
        "Feature2": x2,
        "target": y_noisy,
    }
    df = pd.DataFrame(data)

    print("=" * 80)
    print("Testing NoiseModel...")
    print("=" * 80)

    # Create noise model
    noise_model = NoiseModel(
        df,
        id_column="ID",
        features=["Feature1", "Feature2"],
        target="target",
    )

    # Get noise scores
    print("\nTop 10 noisiest samples:")
    scores = noise_model.get_scores()
    print(scores.head(10))

    # Check if our artificially noisy samples are detected
    noisy_ids = [f"sample_{i}" for i in range(90, 100)]
    detected = scores[scores["ID"].isin(noisy_ids)]
    median_score = scores["noise_score"].median()
    print(f"\nOf 10 noisy samples, {len(detected[detected['noise_score'] > median_score])} above median noise score")

    # Get sample weights
    print("\nSample weights (inverse strategy):")
    weights = noise_model.get_sample_weights(strategy="inverse")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  Mean weight: {weights.mean():.3f}")

    # Get clean subset
    clean = noise_model.get_clean_subset(percentile=90)
    print(f"\nClean subset (bottom 90%): {len(clean)} samples")

    # Get noisy samples
    noisy = noise_model.get_noisy_samples(top_percent=10)
    print(f"\nNoisy samples (top 10%): {len(noisy)} samples")
    print(noisy[["ID", "target", "overfit_residual", "htg_score", "noise_score"]].head())

    # Test with real data
    print("\n" + "=" * 80)
    print("Testing with AQSol data...")
    print("=" * 80)
    fs = FeatureSet("aqsol_features")
    model = Model("aqsol-regression")

    if fs.exists():
        features = model.features()
        target = model.target()
        df = fs.pull_dataframe()

        noise_model = NoiseModel(
            df,
            id_column=fs.id_column,
            features=features,
            target=target,
        )

        print("\nTop 10 noisiest compounds:")
        scores = noise_model.get_scores()
        print(scores.head(10))

        print("\nCoincident conflicts:")
        conflicts = noise_model.coincident_conflicts()
        print(f"Found {len(conflicts)} coincident conflicts")
        if len(conflicts) > 0:
            print(conflicts.head())

        print("\nNoise score distribution:")
        print(scores["noise_score"].describe())
