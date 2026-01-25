"""SmartSample: Intelligently reduce DataFrame rows by aggregating similar rows together."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import logging

# Set up logging
log = logging.getLogger("workbench")


def smart_aggregator(df: pd.DataFrame, target_rows: int = 1000, outlier_column: str = "residual") -> pd.DataFrame:
    """
    Reduce DataFrame rows by aggregating similar rows based on numeric column similarity.

    This is a performant (2-pass) algorithm:
    1. Pass 1: Normalize numeric columns and cluster similar rows using MiniBatchKMeans
    2. Pass 2: Aggregate each cluster (mean for numeric, first for non-numeric)

    Args:
        df: Input DataFrame.
        target_rows: Target number of rows in output (default: 1000).
        outlier_column: Column where high values should resist aggregation (default: "residual").
                       Rows with high values in this column will be kept separate while rows
                       with low values cluster together. Set to None to disable.

    Returns:
        Reduced DataFrame with 'aggregation_count' column showing how many rows were combined.
    """
    if df is None or df.empty:
        return df

    n_rows = len(df)

    # Preserve original column order
    original_columns = df.columns.tolist()

    # If already at or below target, just add the count column and return
    if n_rows <= target_rows:
        result = df.copy()
        result["aggregation_count"] = 1
        return result

    log.info(f"smart_aggregator: Reducing {n_rows} rows to ~{target_rows} rows")

    # Identify columns by type
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        log.warning("smart_aggregator: No numeric columns for clustering, falling back to random sample")
        result = df.sample(n=target_rows)
        result["aggregation_count"] = 1
        return result.reset_index(drop=True)

    # Handle NaN values - fill with column median (use numpy for speed)
    clustering_data = df[numeric_cols].values
    col_medians = np.nanmedian(clustering_data, axis=0)
    nan_mask = np.isnan(clustering_data)
    clustering_data = np.where(nan_mask, col_medians, clustering_data)

    # Normalize and cluster (n_init=1 since MiniBatchKMeans is already approximate)
    X = StandardScaler().fit_transform(clustering_data)
    df["_cluster"] = MiniBatchKMeans(
        n_clusters=min(target_rows, n_rows), random_state=42, batch_size=min(1024, n_rows), n_init=1
    ).fit_predict(X)

    # Post-process: give high-outlier rows their own unique clusters so they don't get aggregated
    if outlier_column and outlier_column in df.columns:
        # Top 10% of outlier values get their own clusters, capped at 20% of target_rows
        n_to_isolate = min(int(n_rows * 0.1), int(target_rows * 0.2))
        outlier_values = df[outlier_column].values
        threshold = np.partition(outlier_values, -n_to_isolate)[-n_to_isolate]
        high_outlier_mask = outlier_values >= threshold
        n_high_outliers = high_outlier_mask.sum()
        # Assign unique cluster IDs starting after the max existing cluster (match dtype to avoid warning)
        max_cluster = df["_cluster"].max()
        new_cluster_ids = np.arange(max_cluster + 1, max_cluster + 1 + n_high_outliers, dtype=df["_cluster"].dtype)
        df.loc[high_outlier_mask, "_cluster"] = new_cluster_ids
        log.info(f"smart_aggregator: Isolated {n_high_outliers} high-outlier rows (>= {threshold:.3f})")
    elif outlier_column:
        log.warning(f"smart_aggregator: outlier_column '{outlier_column}' not found in columns")

    # Aggregate each cluster (mean for numeric, first for non-numeric)
    agg_dict = {col: "mean" for col in numeric_cols} | {col: "first" for col in non_numeric_cols}
    grouped = df.groupby("_cluster")
    result = grouped.agg(agg_dict).reset_index(drop=True)
    result["aggregation_count"] = grouped.size().values

    # Restore original column order, with aggregation_count at the end
    result = result[original_columns + ["aggregation_count"]]

    log.info(f"smart_aggregator: Reduced to {len(result)} rows")
    return result


# Testing
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create test data with clusters
    np.random.seed(42)
    n_samples = 10000

    # Create 3 distinct clusters
    cluster_1 = np.random.randn(n_samples // 3, 3) + np.array([0, 0, 0])
    cluster_2 = np.random.randn(n_samples // 3, 3) + np.array([5, 5, 5])
    cluster_3 = np.random.randn(n_samples // 3, 3) + np.array([10, 0, 5])

    features = np.vstack([cluster_1, cluster_2, cluster_3])

    # Create target and prediction columns, then compute residuals
    target = features[:, 0] + features[:, 1] * 0.5 + np.random.randn(len(features)) * 0.1
    prediction = target + np.random.randn(len(features)) * 0.5  # Add noise for residuals
    residuals = np.abs(target - prediction)

    data = {
        "id": [f"id_{i}" for i in range(len(features))],
        "A": features[:, 0],
        "B": features[:, 1],
        "C": features[:, 2],
        "category": np.random.choice(["cat1", "cat2", "cat3"], len(features)),
        "target": target,
        "prediction": prediction,
        "residual": residuals,
    }
    df = pd.DataFrame(data)

    print(f"Original DataFrame: {len(df)} rows")
    print(df.head())
    print()

    # Test smart_aggregator with residuals preservation
    result = smart_aggregator(df, target_rows=500)
    print(f"smart_aggregator result: {len(result)} rows")
    print(result.head(20))
    print()
    print("Aggregation count stats:")
    print(result["aggregation_count"].describe())
    print()
    # Show that high-residual points have lower aggregation counts
    print("Aggregation count by residual quartile:")
    result["residual_quartile"] = pd.qcut(result["residual"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    print(result.groupby("residual_quartile")["aggregation_count"].mean())

    # Test with real Workbench data
    print("\n" + "=" * 80)
    print("Testing with Workbench data...")
    print("=" * 80)

    from workbench.api import Model

    model = Model("abalone-regression")
    df = model.get_inference_predictions()
    if df is not None:
        print(f"\nOriginal DataFrame: {len(df)} rows")
        print(df.head())

        result = smart_aggregator(df, target_rows=500)
        print(f"\nsmart_aggregator result: {len(result)} rows")
        print(result.head())
        print("\nAggregation count stats:")
        print(result["aggregation_count"].describe())
