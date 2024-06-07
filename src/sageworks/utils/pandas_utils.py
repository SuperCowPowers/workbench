"""Utility/helper methods for Pandas dataframe operations"""

import pandas as pd
from pandas.errors import ParserError
import numpy as np
import json
from io import StringIO
import logging
from typing import Dict, Tuple, List

# SageWorks Logger
log = logging.getLogger("sageworks")


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


def old_examples(df, non_numeric_columns):
    first_n = [df[c].unique()[:5].tolist() if c in non_numeric_columns else ["-"] for c in df.columns]
    first_n = [", ".join([str(x) for x in _list]) for _list in first_n]
    s = pd.Series(first_n, df.columns)
    s.name = "examples"
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


def drop_nans(
    input_df: pd.DataFrame, how: str = "all", nan_drop_percent: float = 50, subset: list = None
) -> pd.DataFrame:
    """Dropping NaNs in rows and columns. Optionally, focus on specific columns.

    Args:
        input_df (pd.DataFrame): Input data frame.
        how (str): 'all' to drop rows where all values are NaN, 'any' to drop rows where any value is NaN.
        nan_drop_percent (float): Percentage threshold to drop columns with missing values exceeding this rate.
        subset (list): Specific subset of columns to check for NaNs when dropping rows.

    Returns:
        pd.DataFrame: DataFrame with NaNs dropped.
    """

    # Grab input number of rows
    orig_num_rows = len(input_df)

    # First replace any INF/-INF with NaN
    output_df = input_df.replace([np.inf, -np.inf], np.nan)

    # Drop Columns that have a large percent of NaNs in them
    column_nan_percent = get_percent_nan(output_df)
    drop_columns = [name for name, percent in column_nan_percent.items() if percent > nan_drop_percent]
    output_df = output_df.drop(drop_columns, axis=1)

    # Report on Dropped Columns
    nan_warn_percent = 0
    for name, percent in column_nan_percent.items():
        if percent > nan_warn_percent:
            log.important(f"Column ({name}) has {percent}% NaN Values")
        if percent > nan_drop_percent:
            log.warning(f"Column ({name}) with {percent}% NaN Values!")

    # Conditionally drop rows based on NaNs in specified columns
    if subset is not None:
        output_df.dropna(axis=0, how=how, subset=subset, inplace=True)
    else:
        output_df.dropna(axis=0, how=how, inplace=True)

    if len(output_df) != orig_num_rows:
        dropped_rows = orig_num_rows - len(output_df)
        log.important(f"Dropping {dropped_rows} rows that have a NaN in them")
        output_df.reset_index(drop=True, inplace=True)

    return output_df


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


def serialize_aws_broker_data(broker_data):
    """
    Serializes a dictionary of DataFrames to a JSON-formatted string.
    Args:
        broker_data (dict): Dictionary of DataFrames
    Returns:
        str: JSON-formatted string
    """
    serialized_dict = {key: df.to_json() for key, df in broker_data.items()}
    return json.dumps(serialized_dict)


def deserialize_aws_broker_data(serialized_data):
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
    from sageworks.utils.test_data_generator import TestDataGenerator

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

    # Get min/max/mean/median/std for numeric columns
    stats_df = numeric_stats(test_df)
    print(stats_df)

    # Clean the DataFrame
    clean_df = drop_nans(test_df)
    log.info(clean_df)

    # Drop Outliers
    norm_df = drop_outliers_iqr(clean_df)
    log.info(norm_df)

    norm_df = drop_outliers_sdev(clean_df)
    log.info(norm_df)
