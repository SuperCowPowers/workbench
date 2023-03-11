"""Utility/helper methods for Pandas dataframe operations"""
import pandas as pd


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
