"""Tests for the FeatureSet Ingest Boundaries (under/overflow, NaN, Inf)"""

import pytest
import pandas as pd
import numpy as np
from workbench.api import FeatureSet
from workbench.core.transforms.pandas_transforms import PandasToFeatures


# Valid subnormal test
def validate_subnormals(subnormals):
    """Test IEEE 754 subnormal numbers"""

    # Check if the values are valid subnormals
    for val in subnormals:
        print(f"Value: {val}")
        if val == 0:
            print("  Invalid: Represents zero, not subnormal.")
        elif val < 4.94e-324 or val >= 2.225e-308:
            print("  Invalid: Out of subnormal range.")
        else:
            print("  Valid: IEEE 754 subnormal.")


@pytest.mark.long
def test_underflow():
    """Underflow Analysis:

    First 5 rows: Above Subnormal Space:
       - Values just above the smallest positive normal number (2.225 x 10^-308).
       - These are fully representable in `float64` with normal precision.
    Last 5 rows: Within Subnormal Space:
       - Values between the smallest positive normal number (2.225 x 10^-308)
         and the smallest positive ^subnormal^ number (4.94 x 10^-324)
       - These are representable but with reduced precision.
    """
    above_subnormal = [2.3e-308, 5e-308, 1e-307, 2e-307, 2.22e-308]
    within_subnormal = [1e-323, 5e-323, 1.5e-323, 2e-323, 4.94e-324]

    # We're going to test that are subnormals are really subnormals
    validate_subnormals(within_subnormal)

    # Create a test DataFrame with above_subnormal and within_subnormal values
    data = {
        "feature1": [42] * 10,  # Control variable :)
        "underflow_feature": above_subnormal + within_subnormal,
        "id": list(range(1, 11)),
    }
    test_df = pd.DataFrame(data)
    print("Test DataFrame:")
    print(test_df)

    # Transform and ingest the dataframe using PandasToFeatures
    feature_set_name = "test_underflow"
    to_features = PandasToFeatures(feature_set_name)
    to_features.set_output_tags(["test", "underflow"])
    to_features.set_input(test_df, id_column="id")
    to_features.transform()

    # Pull the transformed data from the FeatureSet and verify
    fs = FeatureSet(feature_set_name)
    fs_df = fs.pull_dataframe()
    fs_df = fs_df.sort_values(by="id").reset_index(drop=True)  # Sort by ids
    print("FeatureSet DataFrame:")
    print(fs_df)

    # Check for dropped rows
    original_ids = set(test_df["id"])
    ingested_ids = set(fs_df["id"])
    rejected_ids = original_ids - ingested_ids
    print(f"Rejected IDs (due to underflow or ingest errors): {rejected_ids}")


@pytest.mark.long
def test_overflow_nan_inf():
    """Overflow, NaN, and INF Analysis:

    Overflow:
      - Values beyond the range of IEEE 754 double-precision floating-point.
    NaN:
      - Special IEEE 754 value representing "Not a Number."
    INF:
      - Positive and Negative Infinity values.
    """

    # Test data with 8 rows: 4 valid, 4 with special numerics
    data = {
        "overflow_feature": [0, 0, 1e309, -1e309, 0, 0, 0, 0],  # Overflow (both ways)
        "inf_feature": [0, 0, 0, 0, np.inf, -np.inf, 0, 0],  # INF values
        "nan_feature": [0, 0, 0, 0, 0, 0, np.nan, np.nan],  # NaN values
        "id": list(range(1, 9)),  # Unique IDs for each row
    }

    test_df = pd.DataFrame(data)
    print("Test DataFrame:")
    print(test_df)

    # Transform and ingest the dataframe using PandasToFeatures
    feature_set_name = "test_special_values"
    to_features = PandasToFeatures(feature_set_name)
    to_features.set_output_tags(["test", "special"])
    to_features.set_input(test_df, id_column="id")
    to_features.transform()

    # Pull the transformed data from the FeatureSet and verify
    fs = FeatureSet(feature_set_name)
    fs_df = fs.pull_dataframe()
    fs_df = fs_df.sort_values(by="id").reset_index(drop=True)  # Sort by ids
    print("FeatureSet DataFrame:")
    print(fs_df)

    # Check for dropped rows
    original_ids = set(test_df["id"])
    ingested_ids = set(fs_df["id"])
    rejected_ids = original_ids - ingested_ids
    print(f"Rejected IDs (due to overflow, NaN, or INF): {rejected_ids}")


if __name__ == "__main__":
    test_underflow()
    test_overflow_nan_inf()
