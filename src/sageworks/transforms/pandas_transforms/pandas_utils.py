"""Utility/helper methods for Pandas dataframe operations"""
import pandas as pd
import numpy as np

import logging

# SageWorks Imports
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


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


def examples(df, non_numeric_columns):
    first_n = [df[c].unique()[:5].tolist() if c in non_numeric_columns else ["-"] for c in df.columns]
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
    non_numeric_columns = df.select_dtypes(exclude="number").columns.tolist()
    s4 = examples(df, non_numeric_columns)

    # Concatenate the series together
    return pd.concat([s0, s1, s2, s3, s4], axis=1)


def numeric_stats(df):
    """Simple function to get the numeric stats for a dataframe"""
    return df.describe().round(2).T.drop("count", axis=1)


def drop_nans(input_df: pd.DataFrame, how: str = "any", nan_drop_percent: float = 20) -> pd.DataFrame:
    """Dropping NaNs in rows and columns. Obviously lots of ways to do this, so picked some reasonable defaults,
    we can certainly change this later with a more formal set of operations and arguments"""

    # Grab input number of rows
    orig_num_rows = len(input_df)

    # First replace any INF/-INF with NaN
    output_df = input_df.replace([np.inf, -np.inf], np.nan)

    # Drop Columns that have a large percent of NaNs in them
    column_nan_percent = get_percent_nan(output_df)
    drop_columns = [name for name, percent in column_nan_percent.items() if percent > nan_drop_percent]
    output_df = output_df.drop(drop_columns, axis=1)

    # Report on Dropped Columns
    for name, percent in column_nan_percent.items():
        if percent > nan_drop_percent:
            log.warning(f"Dropping Column ({name}) with {percent}% NaN Values!")

    # Drop Rows that have NaNs in them
    output_df.dropna(axis=0, how=how, inplace=True)
    if len(output_df) != orig_num_rows:
        log.info(f"Dropping {orig_num_rows - len(output_df)} rows that have a NaN in them")
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
        scale (float, optional): Scale to use for Standard Deviation. Defaults to 3.0.
    Returns:
        pd.DataFrame: DataFrame with outliers dropped
    """
    # Just the numeric columns
    numeric_df = input_df.select_dtypes(include="number")

    output_df = input_df[numeric_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < sigma).all(axis=1)]
    if input_df.shape[0] != output_df.shape[0]:
        log.info(f"Dropped {input_df.shape[0] - output_df.shape[0]} outlier rows")
    return output_df


if __name__ == "__main__":
    """Exercise the Pandas Utility Methods"""
    from datetime import datetime

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Create some fake data
    fake_data = [
        {
            "id": 1,
            "name": "sue",
            "age": pd.NA,
            "score": 7.8,
            "date": datetime.now(),
            "hobby": pd.NA,
        },
        {
            "id": 2,
            "name": "bob",
            "age": 34,
            "score": pd.NA,
            "date": datetime.now(),
            "hobby": pd.NA,
        },
        {
            "id": 3,
            "name": "ted",
            "age": 69,
            "score": 8.2,
            "date": datetime.now(),
            "hobby": "robots",
        },
        {
            "id": 4,
            "name": "bill",
            "age": pd.NA,
            "score": 7.3,
            "date": datetime.now(),
            "hobby": pd.NA,
        },
        {
            "id": 5,
            "name": "biff",
            "age": 52,
            "score": 7.4,
            "date": datetime.now(),
            "hobby": "robots",
        },
        {
            "id": 6,
            "name": "sally",
            "age": 52,
            "score": 19.5,
            "date": datetime.now(),
            "hobby": "games",
        },
        {
            "id": 7,
            "name": "sammy",
            "age": 52,
            "score": 8.5,
            "date": datetime.now(),
            "hobby": "games",
        },
        {
            "id": 8,
            "name": "jimmy",
            "age": 54,
            "score": 7.5,
            "date": datetime.now(),
            "hobby": "games",
        },
        {
            "id": 9,
            "name": "timmy",
            "age": 53,
            "score": 7.8,
            "date": datetime.now(),
            "hobby": "games",
        },
    ]
    fake_df = pd.DataFrame(fake_data)
    fake_df["name"] = fake_df["name"].astype(pd.StringDtype())
    fake_df["age"] = fake_df["age"].astype(pd.Int64Dtype())
    fake_df["score"] = fake_df["score"].astype(pd.Float64Dtype())
    fake_df["hobby"] = fake_df["hobby"].astype(pd.StringDtype())

    # Get the info about this dataframe
    info_df = info(fake_df)

    # Show the info dataframe
    print(info_df)

    # Get min/max/mean/median/std for numeric columns
    stats_df = numeric_stats(fake_df)
    print(stats_df)

    # Clean the DataFrame
    clean_df = drop_nans(fake_df)
    log.info(clean_df)

    # Drop Outliers
    norm_df = drop_outliers_iqr(clean_df)
    log.info(norm_df)

    norm_df = drop_outliers_sdev(clean_df)
    log.info(norm_df)
