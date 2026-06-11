"""Tests for the DataFrame Store functionality"""

import pandas as pd
import pytest

# Workbench-Bridges Imports
from workbench.api.df_store import DFStore


@pytest.fixture(scope="module")
def df_store():
    return DFStore()


def test_repr(df_store):
    print("DFStore Object...")
    print(df_store)


def test_summary(df_store):
    print("Summary of DataFrames...")
    print(df_store.summary())


def test_details(df_store):
    print("Details of DataFrames...")
    print(df_store.details())


def test_upsert_dataframe(df_store):
    my_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_store.upsert("/tests/test_data", my_df)


def test_upsert_series(df_store):
    series = pd.Series([1, 2, 3, 4], name="Series")
    df_store.upsert("/tests/test_series", series)


def test_get_dataframe(df_store):
    return_value = df_store.get("/tests/test_data")
    print(f"Getting data 'test_data':\n{return_value}")


def test_get_series(df_store):
    return_value = df_store.get("/tests/test_series")
    print(f"Getting data 'test_series':\n{return_value}")


def test_string_types(df_store):
    # Create a DataFrame with int64 values.
    df = pd.DataFrame({"Id": ["A", "B", "C"], "Name": ["Alice", "Bob", "Charlie"]})
    df_store.upsert("/tests/test_string_types", df)
    df_store.delete("/tests/test_string_types")


def test_type_conflation_issues(df_store):
    # Create a DataFrame with int64 values.
    df = pd.DataFrame({"Ver": [123, 456, 789]}, dtype="int64")

    # Sneak in a "-" into one value (now column type becomes 'object').
    df.loc[1, "Ver"] = "-"

    df_store.upsert("/tests/test_type_conflation", df)
    df_store.delete("/tests/test_type_conflation")


def test_deletion(df_store):
    df_store.delete("/tests/test_data")
    df_store.delete("/tests/test_series")


if __name__ == "__main__":

    # Create a DFStore object
    store = DFStore()

    # Run the tests
    test_repr(store)
    test_summary(store)
    test_details(store)
    test_upsert_dataframe(store)
    test_upsert_series(store)
    test_get_dataframe(store)
    test_get_series(store)
    test_repr(store)
    test_string_types(store)
    test_type_conflation_issues(store)
    test_deletion(store)
