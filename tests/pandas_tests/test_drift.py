import pandas as pd
import numpy as np

# Workbench Imports
from workbench.utils.pandas_utils import detect_drift


# Function to create test dataframes (simplified)
def create_test_data(n_rows=100, with_drift=True):
    np.random.seed(42)  # For reproducibility

    # Create base dataframe with different ranges
    current_df = pd.DataFrame(
        {
            "small_range": np.random.uniform(0, 1, n_rows),
            "medium_range": np.random.uniform(10, 100, n_rows),
            "large_range": np.random.uniform(1000, 10000, n_rows),
            "categorical": np.random.choice(["A", "B", "C"], n_rows),
        }
    )

    # Create new dataframe based on drift parameter
    if with_drift:
        # Add small changes to most columns (below threshold)
        new_df = current_df.copy()

        # Add significant drift to medium_range (exceeds threshold)
        # For a 0.1% threshold, this adds a 0.5% change
        column_range = new_df["medium_range"].max() - new_df["medium_range"].min()
        new_df["medium_range"] += column_range * 0.005  # 0.5% drift
    else:
        new_df = current_df.copy()

    return current_df, new_df


# Simple test function
if __name__ == "__main__":
    # Test 1: No drift
    current_df, new_df = create_test_data(with_drift=False)
    has_drift, details = detect_drift(current_df, new_df, drift_percentage=0.1)
    print(f"Test 1 - No drift detected: {not has_drift}")

    # Test 2: With drift
    current_df, new_df = create_test_data(with_drift=True)
    has_drift, details = detect_drift(current_df, new_df, drift_percentage=0.1)
    print(f"Test 2 - Drift detected: {has_drift}")
    print(f"Columns with drift: {details['columns_with_drift']}")

    if not details["drift_examples"].empty:
        print("\nDrift examples (first 3):")
        print(details["drift_examples"].head(3))

    # Test 3: Threshold sensitivity
    has_drift_high, details_high = detect_drift(current_df, new_df, drift_percentage=1.0)
    has_drift_low, details_low = detect_drift(current_df, new_df, drift_percentage=0.01)

    print(f"\nTest 3 - High threshold (1.0%): No drift detected: {not has_drift_high}")
    print(f"Test 3 - Low threshold (0.01%): Drift detected: {has_drift_low}")

    print("\nAll tests completed!")
