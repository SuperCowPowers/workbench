"""Utility/helper methods for Pandas dataframe operations"""
import pandas as pd
import numpy as np


def get_percent_null(df):
    print('Dataframe ({:d} rows)'.format(len(df)))
    s = df.isna().mean().round(3) * 100.0
    s.name = '%Null'
    return s


def unique(df):
    s = df.nunique()
    s.name = '#unique'
    return s


def examples(df):
    first_n = [df[c].unique()[:10].tolist() for c in df.columns]
    s = pd.Series(first_n, df.columns)
    s.name = 'Examples'
    return s


def info(df):
    s1 = get_percent_null(df)
    s2 = unique(df)
    s3 = examples(df)
    df = pd.concat([s1, s2, s3], axis=1)
    return df.sort_values('%Null')


def drop_nans(input_df: pd.DataFrame, how: str = 'any') -> pd.DataFrame:

    # Grab input colums and number of rows
    orig_columns = input_df.columns.tolist()
    orig_num_rows = len(input_df)

    # First replace any INF/-INF with NaN
    output_df = input_df.replace([np.inf, -np.inf], np.nan)

    # Drop Columns that have ALL NaNs in them
    output_df.dropna(axis=1, how='all', inplace=True)
    remaining_columns = output_df.columns.tolist()
    if remaining_columns != orig_columns:
        dropped_columns = list(set(remaining_columns).difference(set(orig_columns)))
        print(f"Dropping {dropped_columns} columns that have a NaN in them")

    # Drop Rows that have NaNs in them
    output_df.dropna(axis=0, how=how, inplace=True)
    if len(output_df) != orig_num_rows:
        print(f"Dropping {orig_num_rows - len(output_df)} rows that have a NaN in them")
        output_df.reset_index(drop=True, inplace=True)

    return output_df
