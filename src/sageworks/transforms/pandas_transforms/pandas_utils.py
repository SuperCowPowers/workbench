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
    log.info('Dataframe ({:d} rows)'.format(len(df)))
    s = df.isna().mean().round(3) * 100.0
    s.name = 'percent_nan'
    return s


def unique(df):
    s = df.nunique()
    s.name = 'num_unique'
    return s


def examples(df):
    first_n = [df[c].unique()[:10].tolist() for c in df.columns]
    s = pd.Series(first_n, df.columns)
    s.name = 'examples'
    return s


def info(df):
    s1 = get_percent_nan(df)
    s2 = unique(df)
    s3 = examples(df)
    info_df = pd.concat([s1, s2, s3], axis=1)
    return info_df.sort_values('percent_nan')


def drop_nans(input_df: pd.DataFrame, how: str = 'any', nan_drop_percent: float = 10) -> pd.DataFrame:
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


def test():
    """Test the Pandas Utility Methods"""
    from datetime import datetime

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Create some fake data
    fake_data = [
        {'id': 1, 'name': 'sue', 'age': pd.NA, 'score': 7.8, 'date': datetime.now(), 'hobby': pd.NA},
        {'id': 2, 'name': 'bob', 'age': 34, 'score': pd.NA, 'date': datetime.now(), 'hobby': pd.NA},
        {'id': 3, 'name': 'ted', 'age': 69, 'score': 8.2, 'date': datetime.now(), 'hobby': 'robots'},
        {'id': 4, 'name': 'bill', 'age': pd.NA, 'score': 5.3, 'date': datetime.now(), 'hobby': pd.NA},
        {'id': 5, 'name': 'sally', 'age': 52, 'score': 9.5, 'date': datetime.now(), 'hobby': 'games'}
    ]
    fake_df = pd.DataFrame(fake_data)

    # Get the info about this dataframe
    info_df = info(fake_df)

    # Show the info dataframe
    log.info(info_df)

    # Clean the DataFrame
    clean_df = drop_nans(fake_df)
    log.info(clean_df)


if __name__ == "__main__":
    test()
