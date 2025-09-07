"""Miscellaneous processing functions for molecular data."""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

# Set up the logger
log = logging.getLogger("workbench")


def geometric_mean(series: pd.Series) -> float:
    """Computes the geometric mean manually to avoid using scipy."""
    return np.exp(np.log(series).mean())


def rollup_experimental_data(
    df: pd.DataFrame, id: str, time: str, target: str, use_gmean: bool = False
) -> pd.DataFrame:
    """
    Rolls up a dataset by selecting the largest time per unique ID and averaging the target value
    if multiple records exist at that time. Supports both arithmetic and geometric mean.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        id (str): Column representing the unique molecule ID.
        time (str): Column representing the time.
        target (str): Column representing the target value.
        use_gmean (bool): Whether to use the geometric mean instead of the arithmetic mean.

    Returns:
        pd.DataFrame: Rolled-up dataframe with all original columns retained.
    """
    # Find the max time per unique ID
    max_time_df = df.groupby(id)[time].transform("max")
    filtered_df = df[df[time] == max_time_df]

    # Define aggregation function
    agg_func = geometric_mean if use_gmean else np.mean

    # Perform aggregation on all columns
    agg_dict = {col: "first" for col in df.columns if col not in [target, id, time]}
    agg_dict[target] = lambda x: agg_func(x) if len(x) > 1 else x.iloc[0]  # Apply mean or gmean

    rolled_up_df = filtered_df.groupby([id, time]).agg(agg_dict).reset_index()
    return rolled_up_df


def micromolar_to_log(series_µM: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of concentrations in µM (micromolar) to their logarithmic values (log10).

    Parameters:
    series_uM (pd.Series): Series of concentrations in micromolar.

    Returns:
    pd.Series: Series of logarithmic values (log10).
    """
    # Replace 0 or negative values with a small number to avoid log errors
    adjusted_series = series_µM.clip(lower=1e-9)  # Alignment with another project

    series_mol_per_l = adjusted_series * 1e-6  # Convert µM/L to mol/L
    log_series = np.log10(series_mol_per_l)
    return log_series


def log_to_micromolar(log_series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series of logarithmic values (log10) back to concentrations in µM (micromolar).

    Parameters:
    log_series (pd.Series): Series of logarithmic values (log10).

    Returns:
    pd.Series: Series of concentrations in micromolar.
    """
    series_mol_per_l = 10**log_series  # Convert log10 back to mol/L
    series_µM = series_mol_per_l * 1e6  # Convert mol/L to µM
    return series_µM


def feature_resolution_issues(df: pd.DataFrame, features: List[str], show_cols: Optional[List[str]] = None) -> None:
    """
    Identify and print groups in a DataFrame where the given features have more than one unique SMILES,
    sorted by group size (largest number of unique SMILES first).

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES strings.
        features (List[str]): List of features to check.
        show_cols (Optional[List[str]]): Columns to display; defaults to all columns.
    """
    # Check for the 'smiles' column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    show_cols = show_cols if show_cols is not None else df.columns.tolist()

    # Drop duplicates to keep only unique SMILES for each feature combination
    unique_df = df.drop_duplicates(subset=[smiles_column] + features)

    # Find groups with more than one unique SMILES
    group_counts = unique_df.groupby(features).size()
    collision_groups = group_counts[group_counts > 1].sort_values(ascending=False)

    # Print each group in order of size (largest first)
    for group, count in collision_groups.items():
        # Get the rows for this group
        if isinstance(group, tuple):
            group_mask = (unique_df[features] == group).all(axis=1)
        else:
            group_mask = unique_df[features[0]] == group

        group_df = unique_df[group_mask]

        print(f"Feature Group (unique SMILES: {count}):")
        print(group_df[show_cols])
        print("\n")


if __name__ == "__main__":
    print("Running molecular processing and transformation tests...")
    print("Note: This requires the molecular_filters module to be available")

    # Test 1: Concentration conversions
    print("\n1. Testing concentration conversions...")

    # Test micromolar to log
    test_conc = pd.Series([1.0, 10.0, 100.0, 1000.0, 0.001])
    log_values = micromolar_to_log(test_conc)
    back_to_uM = log_to_micromolar(log_values)

    print("   µM → log10 → µM:")
    for orig, log_val, back in zip(test_conc, log_values, back_to_uM):
        print(f"   {orig:8.3f} µM → {log_val:6.2f} → {back:8.3f} µM")

    # Test 2: Geometric mean
    print("\n2. Testing geometric mean...")
    test_series = pd.Series([2, 4, 8, 16])
    geo_mean = geometric_mean(test_series)
    arith_mean = np.mean(test_series)
    print(f"   Series: {list(test_series)}")
    print(f"   Arithmetic mean: {arith_mean:.2f}")
    print(f"   Geometric mean: {geo_mean:.2f}")

    # Test 3: Experimental data rollup
    print("\n3. Testing experimental data rollup...")

    # Create test data with multiple timepoints and replicates
    test_data = pd.DataFrame(
        {
            "compound_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "time": [1, 2, 2, 1, 2, 1, 1, 2],
            "activity": [10, 20, 22, 5, 8, 100, 110, 200],
            "assay": ["kinase", "kinase", "kinase", "kinase", "kinase", "cell", "cell", "cell"],
        }
    )

    # Rollup with arithmetic mean
    rolled_arith = rollup_experimental_data(test_data, "compound_id", "time", "activity", use_gmean=False)
    print("   Arithmetic mean rollup:")
    print(rolled_arith[["compound_id", "time", "activity"]])

    # Rollup with geometric mean
    rolled_geo = rollup_experimental_data(test_data, "compound_id", "time", "activity", use_gmean=True)
    print("\n   Geometric mean rollup:")
    print(rolled_geo[["compound_id", "time", "activity"]])

    # Test 4: Feature resolution issues
    print("\n4. Testing feature resolution identification...")

    # Create data with some duplicate features but different SMILES
    resolution_df = pd.DataFrame(
        {
            "smiles": ["CCO", "C(C)O", "CC(C)O", "CCC(C)O", "CCCO"],
            "assay_id": ["A1", "A1", "A2", "A2", "A3"],
            "value": [1.0, 1.5, 2.0, 2.2, 3.0],
        }
    )

    print("   Checking for feature collisions in 'assay_id':")
    feature_resolution_issues(resolution_df, ["assay_id"], show_cols=["smiles", "assay_id", "value"])

    # Test 7: Edge cases
    print("\n7. Testing edge cases...")

    # Zero and negative concentrations
    edge_conc = pd.Series([0, -1, 1e-10])
    edge_log = micromolar_to_log(edge_conc)
    print("   Edge concentration handling:")
    for c, l in zip(edge_conc, edge_log):
        print(f"      {c:6.2e} µM → {l:6.2f}")

    print("\n✅ All molecular processing tests completed!")
