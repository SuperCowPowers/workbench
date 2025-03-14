"""Utility/helper methods for Pandas dataframe operations"""

import pandas as pd
from pandas.errors import ParserError
import numpy as np
import sys
import json
from io import StringIO
import hashlib
import logging
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import confusion_matrix

# Workbench Logger
log = logging.getLogger("workbench")


class DataFrameBuilder:
    def __init__(self):
        self.rows = []

    def add_row(self, row_data: dict):
        """Adds a new row to the DataFrame.

        Parameters:
            row_data (dict): Key-value pairs representing column names and their values for the row.
        """
        self.rows.append(row_data)

    def build(self) -> pd.DataFrame:
        """Constructs the DataFrame from the accumulated rows.

        Returns:
            A pandas DataFrame containing all the added rows.
        """
        return pd.DataFrame(self.rows)


def df_hash(df: pd.DataFrame) -> str:
    """Calculate a hash for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to hash.

    Returns:
        str: Hash of the DataFrame.
    """
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def dataframe_delta(func_that_returns_df, previous_hash: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """Generalized method to compute the delta for any function that returns a DataFrame.

    Args:
        func_that_returns_df (callable): A function that returns a DataFrame.
        previous_hash (str): Optional hash of the previous DataFrame for change detection.

    Returns:
        tuple: (DataFrame, current_hash)
               - DataFrame is None if there are no changes, otherwise the updated DataFrame.
               - current_hash is the hash of the current DataFrame.
    """
    # Get the current DataFrame
    df = func_that_returns_df()

    # Compute the hash for the DataFrame
    current_hash = df_hash(df)

    # Compare hashes
    if current_hash == previous_hash:
        return None, current_hash

    # Return the DataFrame and the new hash if changed
    return df, current_hash


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, display_columns: list):
    """Compare two DataFrames and report on differences.

    Args:
        df1 (pd.DataFrame): First DataFrame to compare.
        df2 (pd.DataFrame): Second DataFrame to compare.
        display_columns (list): Columns to display when differences are found.
    """

    # Check if the entire dataframes are equal
    if df1.equals(df2):
        print("The DataFrames are equal :)")
        return

    # Report on column differences
    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)
    only_in_df1 = df1_columns - df2_columns
    only_in_df2 = df2_columns - df1_columns
    if only_in_df1:
        print("\nColumns only in df1:", only_in_df1)
    if only_in_df2:
        print("\nColumns only in df2:", only_in_df2)

    # For common columns make sure they all have the same data types
    common_columns = list(df1_columns.intersection(df2_columns))
    same_types = True
    for column in common_columns:
        if df1[column].dtype != df2[column].dtype:
            same_types = False
            print(f"\nColumn '{column}' has different data types: {df1[column].dtype} vs {df2[column].dtype}")
    if same_types:
        print("\nAll common columns have the same data types")
        # Print out the column types
        print("\nColumn Types:")
        print(f"DF1: {df1[common_columns].dtypes.value_counts()}")
        print(f"DF2: {df2[common_columns].dtypes.value_counts()}")

    # Count the NaNs in each DataFrame individually (only show columns with > 0 NaNs)
    nan_counts_df1 = df1.isna().sum()
    nan_counts_df2 = df2.isna().sum()
    nan_counts_df1 = nan_counts_df1[nan_counts_df1 > 0]
    nan_counts_df2 = nan_counts_df2[nan_counts_df2 > 0]
    if not nan_counts_df1.empty:
        print("\nNaN Counts in df1:")
        print(nan_counts_df1)
    if not nan_counts_df2.empty:
        print("\nNaN Counts in df2:")
        print(nan_counts_df2)

    # Define tolerance for float comparisons
    epsilon = 1e-10

    # Check for differences in common columns
    for column in common_columns:
        if pd.api.types.is_string_dtype(df1[column]) or pd.api.types.is_string_dtype(df2[column]):
            # String comparison with NaNs treated as equal
            differences = ~(df1[column].fillna("") == df2[column].fillna(""))
        elif pd.api.types.is_float_dtype(df1[column]) or pd.api.types.is_float_dtype(df2[column]):
            # Float comparison within epsilon with NaNs treated as equal
            differences = ~((df1[column] - df2[column]).abs() <= epsilon) & ~(
                pd.isna(df1[column]) & pd.isna(df2[column])
            )
        else:
            # Other types (e.g., int) with NaNs treated as equal
            differences = ~(df1[column].fillna(0) == df2[column].fillna(0))

        # Create a merged DataFrame showing values from both DataFrames
        merged_df = pd.DataFrame(
            {
                **{col: df1.loc[differences, col] for col in display_columns},
                f"{column}_1": df1.loc[differences, column],
                f"{column}_2": df2.loc[differences, column],
            }
        )

        # If differences exist, display them
        if differences.any():
            print(f"\nColumn {column} has differences:")

            # Create a merged DataFrame showing values from both DataFrames
            merged_df = pd.DataFrame(
                {
                    **{col: df1.loc[differences, col] for col in display_columns},
                    f"{column}_1": df1.loc[differences, column],
                    f"{column}_2": df2.loc[differences, column],
                }
            )

            # Display the merged DataFrame
            print(merged_df)


def subnormal_check(df):
    """
    Identifies and prints rows from the DataFrame containing subnormal floats.

    A subnormal float is smaller than the smallest positive normal float
    (~2.2e-308) but greater than zero (~5e-324).
    """
    # Define the range for subnormal floats
    min_positive = np.nextafter(0, 1)  # Smallest positive subnormal float (~5e-324)
    min_normal = sys.float_info.min  # Smallest positive normal float (~2.2e-308)

    # Check for subnormal floats in the DataFrame
    subnormal_mask = df.map(lambda x: isinstance(x, float) and min_positive <= abs(x) < min_normal)

    # Get rows where any column contains a subnormal float
    rows_with_subnormal = df[subnormal_mask.any(axis=1)]

    if not rows_with_subnormal.empty:
        print("Rows containing subnormal floats:")
        print(rows_with_subnormal)
    else:
        print("No subnormal floats found in the DataFrame.")


def binary_accuracy(conf_matrix: pd.DataFrame, positive_classes: list, negative_classes: list) -> float:
    """
    Computes binary accuracy from a confusion matrix.

    Parameters:
        conf_matrix (pd.DataFrame): Confusion matrix as a DataFrame.
        positive_classes (list): List of class labels considered as 'positive'.
        negative_classes (list): List of class labels considered as 'negative'.

    Returns:
        float: Binary accuracy score.
    """
    # True Positives (TP) - Correctly classified as Positive
    tps = conf_matrix.loc[positive_classes, positive_classes].values.sum()

    # True Negatives (TN) - Correctly classified as Negative
    tns = conf_matrix.loc[negative_classes, negative_classes].values.sum()

    # False Positives (FP) - Negative misclassified as Positive
    fps = conf_matrix.loc[negative_classes, positive_classes].values.sum()

    # False Negatives (FN) - Positive misclassified as Negative
    fns = conf_matrix.loc[positive_classes, negative_classes].values.sum()

    # Compute Binary Accuracy
    accuracy = (tps + tns) / (tps + tns + fps + fns)
    return accuracy


def split_dataframe_by_quantiles(df: pd.DataFrame, column: str, quantiles: int = 10) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into N quantile-based DataFrames based on the specified column.
    Ensures highest values are in the first quantiles and prevents gaps in quantile indices.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column used for quantile splitting.
        quantiles (int): The number of quantiles to split into (default is 10).

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing a quantile.
    """
    # Compute quantiles, reversing order so highest values get lower indices
    df["quantile"] = pd.qcut(df[column], q=quantiles, labels=False, duplicates="drop")

    # If fewer than expected quantiles exist, reindex to ensure 0-based sequential bins
    unique_quantiles = sorted(df["quantile"].dropna().unique(), reverse=True)
    remap = {old: new for new, old in enumerate(unique_quantiles)}
    df["quantile"] = df["quantile"].map(remap)

    # Create a list of DataFrames for each quantile
    quantile_dfs = [df[df["quantile"] == i].drop(columns=["quantile"]) for i in range(len(unique_quantiles))]
    return quantile_dfs


def max_proba(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'max_proba' to the DataFrame, which is the max value
    among all columns ending in '_proba'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The updated DataFrame with the new 'max_proba' column.
    """
    # Identify probability columns
    proba_cols = [col for col in df.columns if col.endswith("_proba")]

    # Compute max probability per row
    df["max_proba"] = df[proba_cols].max(axis=1)
    return df


def confidence_profile(
    df: pd.DataFrame, target: str, positive_classes: list, negative_classes: list, quantiles: int = 10
) -> pd.DataFrame:
    """
    Computes binary accuracy at different confidence quantiles.

    Parameters:
        df (pd.DataFrame): DataFrame containing predictions and probabilities.
        target (str): Name of the target column.
        positive_classes (list): List of class labels considered as 'positive'.
        negative_classes (list): List of class labels considered as 'negative'.
        quantiles (int): Number of quantiles to split data into (default = 10).

    Returns:
        pd.DataFrame: A DataFrame with confidence quantiles and corresponding binary accuracy.
    """
    # Step 1: Compute max confidence per row
    df = max_proba(df)

    # Step 2: Split dataframe into quantiles based on max confidence
    quantile_dfs = split_dataframe_by_quantiles(df, "max_proba", quantiles)

    # Step 3: Compute accuracy per quantile
    quant_accuracy = []

    for i, q_df in enumerate(quantile_dfs):
        # Step 3a: Create confusion matrix for this quantile
        cm = pd.DataFrame(
            confusion_matrix(q_df[target], q_df["prediction"], labels=positive_classes + negative_classes),
            index=positive_classes + negative_classes,
            columns=positive_classes + negative_classes,
        )

        # Step 3b: Compute binary accuracy
        accuracy = binary_accuracy(cm, positive_classes, negative_classes)

        # Step 3c: Store result
        quant_accuracy.append({"quantile": i, "binary_accuracy": accuracy})

    # Convert results to DataFrame for easy plotting
    accuracy_df = pd.DataFrame(quant_accuracy)
    return accuracy_df


def temporal_split(df: pd.DataFrame, date_column: str, test_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing sets based on chronological order,
    handling 'clumps' (dates with a large number of rows) separately.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): The name of the datetime column.
        test_split (float): Proportion of data to be used for testing (default 0.2).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])  # Ensure datetime type

    # Compute clump threshold (5% of total rows, min 10 rows)
    clump_threshold = max(int(len(df) * 0.05), 10)

    # Count occurrences per date
    date_counts = df[date_column].value_counts()

    # Identify clump and non-clump dates
    clump_dates = date_counts[date_counts > clump_threshold].index
    non_clump_dates = date_counts[date_counts <= clump_threshold].index

    # Separate clump and non-clump data
    clump_df = df[df[date_column].isin(clump_dates)]
    non_clump_df = df[df[date_column].isin(non_clump_dates)]

    # Handle clump splits (80/20 within each clump)
    train_clumps, test_clumps = [], []
    for date in clump_dates:
        subset = clump_df[clump_df[date_column] == date]
        train_size = int(len(subset) * (1 - test_split))
        train_clumps.append(subset.iloc[:train_size])
        test_clumps.append(subset.iloc[train_size:])

    # Concatenate clump splits
    train_clump_df = pd.concat(train_clumps) if train_clumps else pd.DataFrame(columns=df.columns)
    test_clump_df = pd.concat(test_clumps) if test_clumps else pd.DataFrame(columns=df.columns)

    # Handle non-clump splits (sorted by date, then split)
    sorted_non_clump_df = non_clump_df.sort_values(by=date_column)
    split_index = int(len(sorted_non_clump_df) * (1 - test_split))
    train_non_clump_df = sorted_non_clump_df.iloc[:split_index]
    test_non_clump_df = sorted_non_clump_df.iloc[split_index:]

    # Final train/test DataFrames
    train_df = pd.concat([train_clump_df, train_non_clump_df])
    test_df = pd.concat([test_clump_df, test_non_clump_df])

    return train_df, test_df


def get_percent_nan(df):
    log.info("DataFrame ({:d} rows)".format(len(df)))
    s = df.isna().mean().round(3) * 100.0
    s.name = "percent_nan"
    return s


def unique(df):
    s = df.nunique()
    s.name = "num_unique"
    return s


def column_dtypes(df):
    s = df.dtypes
    s.name = "dtype"
    return s


def examples(df):
    first_n = [df[c].unique()[:5].tolist() for c in df.columns]
    first_n = [", ".join([str(x) for x in _list]) for _list in first_n]
    s = pd.Series(first_n, df.columns)
    s.name = "examples"
    return s


def info(df):
    # Get the number of unique values for each column
    s0 = column_dtypes(df)
    s1 = df.count()
    s1.name = "count"
    s2 = get_percent_nan(df)
    s3 = unique(df)

    # Remove all the numeric columns from the original dataframe
    # non_numeric_columns = df.select_dtypes(exclude="number").columns.tolist()
    s4 = examples(df)

    # Concatenate the series together
    return pd.concat([s0, s1, s2, s3, s4], axis=1)


def numeric_stats(df):
    """Simple function to get the numeric stats for a dataframe"""
    return df.describe().round(2).T.drop("count", axis=1)


def convert_to_numeric(df, verbose=False) -> pd.DataFrame:
    """Converts all columns in a DataFrame to numeric, reporting parse errors and coercing invalid values to NaN.

    Args:
        df (pd.DataFrame): Input DataFrame.
        verbose (bool): Whether to log failed parsing attempts.

    Returns:
        pd.DataFrame: DataFrame with columns converted to numeric.
    """

    # Backup the original DataFrame before conversion
    original_df = df.copy()

    # Convert to numeric, coercing invalid values to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # If verbose, find locations where values were coerced to NaN and report
    if verbose:
        failed_parsing = df.isna() & ~original_df.isna()
        for column in failed_parsing.columns:
            if failed_parsing[column].any():
                failed_rows = original_df.loc[failed_parsing[column], column]
                for index, value in failed_rows.items():
                    log.warning(f"Failed to parse value '{value}' in column '{column}' at index {index}.")

    # Return the DataFrame with columns converted to numeric
    return df


def remove_rows_with_nans(input_df: pd.DataFrame, how: str = "any", subset: list = None) -> pd.DataFrame:
    """
    Removes rows with NaN or INF values from the DataFrame.

    Args:
        input_df (pd.DataFrame): Input data frame.
        how (str): 'any' to drop rows where any value is NaN/INF, 'all' to drop rows where all values are NaN/INF.
        subset (list): Specific subset of columns to check for NaN/INF.

    Returns:
        pd.DataFrame: DataFrame with rows containing NaN/INF removed.
    """
    log.info("Replacing INF/-INF values with NaN.")
    input_df = input_df.replace([np.inf, -np.inf], np.nan)

    orig_num_rows = len(input_df)
    log.info(f"Checking for rows with NaN values using '{how}' condition.")

    input_df = input_df.dropna(axis=0, how=how, subset=subset)
    dropped_rows = orig_num_rows - len(input_df)

    if dropped_rows > 0:
        log.warning(f"Dropped {dropped_rows} rows with NaN or INF values.")
    else:
        log.info("No rows with NaN or INF values found.")

    input_df.reset_index(drop=True, inplace=True)
    return input_df


def feature_quality_metrics(input_df: pd.DataFrame, feature_list: list, strategy: str = "median") -> pd.DataFrame:
    """
    Imputes NaN/INF values in the specified feature columns of the DataFrame and tracks feature quality issues.

    Args:
        input_df (pd.DataFrame): Input data frame.
        feature_list (list): List of feature columns to process.
        strategy (str): Imputation strategy. Currently supports 'mean', 'median', or 'zero'.

    Returns:
        pd.DataFrame: DataFrame with NaN/INF values imputed and a 'feature_quality_tags' column.

    Behavior:
        1. Tracks and tags rows with feature quality issues:
            - 'parse': Values that failed to parse when converting to numeric.
            - 'inf': INF/-INF values in the DataFrame.
            - 'nan': NaN values identified during processing.
        2. Replaces parse errors and inf values with NaN.
        3. Imputes all NaN values using the specified strategy ('mean', 'median', or 'zero').
        4. Logs detailed information about detected and handled feature quality issues.
    """
    # Backup the original DataFrame and isolate feature columns for processing
    process_df = input_df[feature_list].copy()

    # Initialize the feature_quality_tags column as an empty list for all rows
    feature_quality_tags = [set() for _ in range(len(process_df))]

    # Convert feature columns to numeric, coercing invalid values to NaN
    process_df = process_df.apply(pd.to_numeric, errors="coerce")

    # Track and tag parsing errors
    parsing_error_counts = defaultdict(lambda: defaultdict(int))

    # Get indices of failed parsing as a single DataFrame
    failed_parsing = process_df.isna() & ~input_df[feature_list].isna()
    failed_indices = failed_parsing.stack()  # Stack to get a Series of True values with multi-index (row, column)

    # Filter to keep only True (failed parsing) entries
    failed_indices = failed_indices[failed_indices]

    # Process all parsing errors at once
    for row, column in failed_indices.index:
        value = input_df.at[row, column]
        feature_quality_tags[row].add("parse")
        parsing_error_counts[column][str(value)] += 1

    # Report parsing error counts
    for column, errors in parsing_error_counts.items():
        for value, count in errors.items():
            log.important(f"Column '{column}' had {count} occurrences of the parsing error value '{value}'.")

    # Count and log the number of INF/-INF values
    inf_locations = np.isinf(process_df[feature_list])
    inf_count = inf_locations.sum().sum()  # Total count across all rows and columns

    if inf_count > 0:
        for column in process_df.columns:
            if inf_locations[column].any():
                inf_rows = process_df.loc[inf_locations[column], column]
                for index, value in inf_rows.items():
                    log.important(f"'{value}' found in column '{column}' at index {index}.")
                    feature_quality_tags[index].add("inf")

        # Replace INF/-INF values with NaN
        process_df = process_df.replace([np.inf, -np.inf], np.nan)

    # Log and handle NaN values
    log.info(f"Imputing missing values using '{strategy}' strategy.")
    for col in process_df.columns:
        if process_df[col].isna().any():
            # Determine the imputation value
            if strategy == "mean":
                fill_value = process_df[col].mean()
            elif strategy == "median":
                fill_value = process_df[col].median()
            elif strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unsupported imputation strategy: {strategy}")

            # Count the number of NaN values to be replaced
            num_values = process_df[col].isna().sum()

            # Update the feature_quality_tags for NaN values
            nan_indices = process_df[process_df[col].isna()].index
            for index in nan_indices:
                feature_quality_tags[index].add("nan")

            # Update the column
            process_df[col] = process_df[col].fillna(fill_value)
            log.important(f"Imputing {col} replacing {num_values} values with {strategy}({fill_value:.2f})")

    # Combine processed feature columns with untouched columns and feature_quality_tags
    untouched_columns = input_df.drop(columns=feature_list)
    result_df = pd.concat([untouched_columns, process_df], axis=1)
    result_df["feature_quality_tags"] = [list(tags) for tags in feature_quality_tags]

    return result_df


def drop_duplicates(input_df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from a dataframe
    Args:
        input_df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with duplicate rows dropped
    """

    # Drop Duplicates
    output_df = input_df.drop_duplicates()
    if input_df.shape[0] != output_df.shape[0]:
        log.info(f"Dropped {input_df.shape[0] - output_df.shape[0]} duplicate rows")
    return output_df


def drop_outliers_iqr(input_df: pd.DataFrame, scale: float = 1.5) -> pd.DataFrame:
    """Drop outliers from a dataframe
    Args:
        input_df (pd.DataFrame): Input DataFrame
        scale (float, optional): Scale to use for IQR. Defaults to 1.5.
    Returns:
        pd.DataFrame: DataFrame with outliers dropped
    """

    # Just the numeric columns
    numeric_df = input_df.select_dtypes(include="number")

    # Drop Outliers using IQR
    q1 = numeric_df.quantile(0.25, numeric_only=True)
    q3 = numeric_df.quantile(0.75, numeric_only=True)
    iqr_scale = (q3 - q1) * scale
    output_df = input_df[~((numeric_df < (q1 - iqr_scale)) | (numeric_df > (q3 + iqr_scale))).any(axis=1)]
    if input_df.shape[0] != output_df.shape[0]:
        log.info(f"Dropped {input_df.shape[0] - output_df.shape[0]} outlier rows")
    return output_df


def drop_outliers_sdev(input_df: pd.DataFrame, sigma: float = 2.0) -> pd.DataFrame:
    """Drop outliers from a dataframe
    Args:
        input_df (pd.DataFrame): Input DataFrame
        sigma (float, optional): Scale to use for Standard Deviation. Defaults to 2.0.
    Returns:
        pd.DataFrame: DataFrame with outliers dropped
    """
    # Just the numeric columns
    numeric_df = input_df.select_dtypes(include="number")

    output_df = input_df[numeric_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < sigma).all(axis=1)]
    if input_df.shape[0] != output_df.shape[0]:
        log.info(f"Dropped {input_df.shape[0] - output_df.shape[0]} outlier rows")
    return output_df


def shorten_values(df: pd.DataFrame, max_length: int = 100) -> pd.DataFrame:
    """This method should be used with caution"""

    def truncate_element(element):
        if isinstance(element, str) and len(element) > max_length:
            return element[:max_length] + "..."  # Add ellipsis to indicate truncation
        return element

    return df.map(truncate_element)


def subplot_positions(df: pd.DataFrame, num_columns: int = 2) -> Dict[str, Tuple[int, int]]:
    positions = {}
    for i, col in enumerate(df.columns):
        row = i // num_columns + 1
        col_pos = i % num_columns + 1
        positions[col] = (row, col_pos)
    return positions


def displayable_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """Experimental: Create a displayable dataframe from FeatureSet data
    Args:
        input_df (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: DataFrame with displayable columns
    """
    exclude_columns = ["write_time", "api_invocation_time", "is_deleted", "training"]
    df = input_df[input_df.columns.difference(exclude_columns)].copy()
    dummy_cols = get_dummy_cols(df)

    # Okay, so this is a bit of a hack, but we need to replace all but the last underscore
    # run the from_dummies method, then change the column names back to the original
    un_dummy = undummify(df[dummy_cols])
    return pd.concat([df.drop(dummy_cols, axis=1), un_dummy], axis=1)


def undummify(df, prefix_sep="_"):
    """Experimental: Undummify a dataframe"""
    cols2collapse = {prefix_sep.join(item.split(prefix_sep)[:-1]): (prefix_sep in item) for item in df.columns}
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = df.filter(like=col).idxmax(axis=1).apply(lambda x: x.split(prefix_sep)[-1]).rename(col)
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def get_dummy_cols(df: pd.DataFrame) -> list:
    """Determines a list of dummy columns for the given DataFrame
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        list: List of dummy columns
    """
    dum_cols = list(df.select_dtypes(include=["int", "bool"]).columns)
    underscore_cols = [col for col in df.columns if "_" in col and col in dum_cols]
    dummy_cols = []
    for col in underscore_cols:
        # Just columns with 0 and 1
        if set(df[col].unique()).issubset([0, 1]):
            dummy_cols.append(col)
    return dummy_cols


def athena_to_pandas_types(df: pd.DataFrame, column_athena_types: dict) -> pd.DataFrame:
    """Converts a dataframe into the proper Pandas types
    Args:
        df (pd.DataFrame): The DataFrame we want to convert types for
        column_athena_types (dict): A dictionary of Athena types for each column
    Returns:
        pd.DataFrame: The DataFrame with the proper types
    """

    # Sanity check
    if df.empty:
        return df

    # Convert the Athena types to Pandas types with this mapper
    athena_to_pandas_mapper = {
        "tinyint": "Int8",
        "smallint": "Int16",
        "int": "Int32",
        "integer": "Int32",
        "bigint": "Int64",
        "boolean": "boolean",
        "float": "float32",
        "double": "float64",
        "real": "float64",
        "decimal": "float64",
        "numeric": "float64",
        "char": "string",
        "varchar": "string",
        "string": "string",
        "date": "datetime64[ns]",
        "timestamp": "datetime64[ns]",
        "binary": "object",
        "array": "object",
        "map": "object",
        "struct": "object",
        "uniontype": "object",
    }

    # Map the Athena types to Pandas types
    pandas_column_types = {
        col: athena_to_pandas_mapper[athena_type] for col, athena_type in column_athena_types.items()
    }

    # Convert the DataFrame columns to the mapped Pandas types
    df = df.astype(pandas_column_types)
    return df


def convert_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to automatically convert object columns to something more concrete"""
    for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
        # Try to convert object to datetime
        if "date" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError, TypeError):
                log.debug(f"Column {c} could not be converted to datetime...")

        # Try to convert object to string
        else:
            try:
                df[c] = df[c].astype(str)
            except (ParserError, ValueError, TypeError):
                log.debug(f"Column {c} could not be converted to string...")
    return df


def serialize_aws_metadata(metadata):
    """
    Serializes a dictionary of DataFrames to a JSON-formatted string.
    Args:
        metadata (dict): Dictionary of DataFrames
    Returns:
        str: JSON-formatted string
    """
    serialized_dict = {key: df.to_json() for key, df in metadata.items()}
    return json.dumps(serialized_dict)


def deserialize_aws_metadata(serialized_data):
    """
    Deserializes a JSON-formatted string to a dictionary of DataFrames.
    Args:
        serialized_data (str): JSON-formatted string
    Returns:
        dict: Dictionary of DataFrames
    """
    deserialized_dict = json.loads(serialized_data)
    return {key: pd.read_json(StringIO(df_json)) for key, df_json in deserialized_dict.items()}


def expand_proba_column(df: pd.DataFrame, class_labels: List[str]) -> pd.DataFrame:
    """
    Expands a column in a DataFrame containing a list of probabilities into separate columns.

    Args:
        df (pd.DataFrame): DataFrame containing a "pred_proba" column
        class_labels (List[str]): List of class labels

    Returns:
        pd.DataFrame: DataFrame with the "pred_proba" expanded into separate columns
    """

    # Sanity check
    proba_column = "pred_proba"
    if proba_column not in df.columns:
        raise ValueError('DataFrame does not contain a "pred_proba" column')

    # Construct new column names with '_proba' suffix
    new_col_names = [f"{label}_proba" for label in class_labels]

    # Expand the proba_column into separate columns for each probability
    proba_df = pd.DataFrame(df[proba_column].tolist(), columns=new_col_names)

    # Drop the original proba_column and reset the index in prep for the concat
    df = df.drop(columns=[proba_column])
    df = df.reset_index(drop=True)

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, proba_df], axis=1)
    return df


def stratified_split(df, column_name, test_size=0.2, random_state=42):
    """
    Stratified train-test split based on a categorical column, including handling NaNs as a separate category.

    Args:
        df (pd.DataFrame): DataFrame to split
        column_name (str): Column name to stratify the split on
        test_size (float): Proportion of the test set
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame, pd.DataFrame: Train and Test DataFrames

    """
    # Temporarily replace NaNs with a placeholder to treat them as a category
    df_temp = df.copy()
    df_temp.loc[:, column_name] = df_temp[column_name].fillna("NaN")  # Use .loc to avoid SettingWithCopyWarning

    # Determine minimum number of samples per group in the test set
    min_test_samples = 1

    # Calculate the number of samples each group should ideally have in the test set
    group_counts = df_temp[column_name].value_counts()

    # This ensures at least one sample per group
    test_counts = (group_counts * test_size).clip(lower=min_test_samples).astype(int)

    # Create a mask to identify test samples
    test_mask = pd.Series(False, index=df_temp.index)

    # Assign samples to test set ensuring each group has at least one sample
    grouped = df_temp.groupby(column_name)
    for name, group in grouped:
        test_indices = group.sample(n=test_counts[name], random_state=random_state).index
        test_mask.loc[test_indices] = True

    train_df = df_temp[~test_mask]
    test_df = df_temp[test_mask]

    # Convert 'NaN' placeholders back to actual NaNs, using .loc to ensure direct modification
    train_df.loc[:, column_name] = train_df[column_name].replace("NaN", np.nan)
    test_df.loc[:, column_name] = test_df[column_name].replace("NaN", np.nan)

    return train_df, test_df


if __name__ == "__main__":
    """Exercise the Pandas Utility Methods"""
    from workbench.utils.test_data_generator import TestDataGenerator

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Generate some test data
    test_data = TestDataGenerator()
    test_df = test_data.person_data()

    # Get the info about this dataframe
    info_df = info(test_df)

    # Show the info dataframe
    print(info_df)

    # Get the hash of the DataFrame
    hash_value = df_hash(test_df)
    print(hash_value)

    # Get min/max/mean/median/std for numeric columns
    stats_df = numeric_stats(test_df)
    print(stats_df)

    # Clean the DataFrame
    clean_df = remove_rows_with_nans(test_df)
    log.info(clean_df)

    # Drop Outliers
    norm_df = drop_outliers_iqr(clean_df)
    log.info(norm_df)

    norm_df = drop_outliers_sdev(clean_df)
    log.info(norm_df)

    # Create two dataframes for comparison
    my_df1 = test_data.person_data()
    my_df2 = my_df1.copy()
    compare_dataframes(my_df1, my_df2, ["Id", "Name"])

    # Make a couple of differences
    my_df2.loc[6, "Height"] = 99
    my_df2.loc[48, "Salary"] = 99000
    my_df2.loc[82, "Age"] = 99
    my_df2["new_column"] = "new_value"
    compare_dataframes(my_df1, my_df2, ["Id", "Name"])

    # Subnormal Check
    data = {"col1": [5, 0, 1e-310, 5e-325], "col2": [3, 0, 1e-100, -1e-320]}
    df = pd.DataFrame(data)
    subnormal_check(df)

    # Example for Binary Accuracy
    conf_matrix = pd.DataFrame(
        {
            "bad_low": [11, 1, 0, 0, 0],
            "okay_low": [19, 413, 64, 0, 0],
            "good": [0, 92, 1161, 111, 17],
            "okay_high": [0, 0, 42, 188, 34],
            "bad_high": [0, 0, 8, 18, 96],
        },
        index=["bad_low", "okay_low", "good", "okay_high", "bad_high"],
    )

    # Define Positive and Negative Classes
    positive_classes = ["okay_low", "good", "okay_high"]
    negative_classes = ["bad_low", "bad_high"]

    # Compute Binary Accuracy
    accuracy = binary_accuracy(conf_matrix, positive_classes, negative_classes)
    print(f"Binary Accuracy: {accuracy:.3f}")

    # Test split_dataframe_by_quantiles
    data = {"id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    df = pd.DataFrame(data)
    quantile_dfs = split_dataframe_by_quantiles(df, "value", quantiles=3)
    for i, quantile_df in enumerate(quantile_dfs):
        print(f"Quantile {i + 1}:")
        print(quantile_df)

    # Test confidence_profile
    data = {
        "class": ["low", "low", "high", "med", "low", "high", "low", "high", "low", "high"],
        "prediction": ["low", "low", "high", "high", "low", "high", "low", "high", "low", "med"],
        "pred_proba": [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
            [0.6, 0.2, 0.2],
            [0.1, 0.3, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
            [0.25, 0.5, 0.25],
        ],
    }
    df = pd.DataFrame(data)
    df = expand_proba_column(df, ["low", "med", "high"])
    accuracy_df = confidence_profile(df, "class", ["high"], ["med", "low"])
    print(accuracy_df)

    # Add a datetime column to our df and test temporal_split
    df["date"] = pd.date_range(start="1/1/2020", periods=len(df))
    train_df, test_df = temporal_split(df, "date", test_split=0.2)
    print("Train DataFrame:")
    print(train_df)
    print("\nTest DataFrame:")
    print(test_df)
