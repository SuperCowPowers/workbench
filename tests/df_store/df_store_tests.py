"""Tests for the DataFrame Store functionality"""

import pandas as pd

# Workbench-Bridges Imports
from workbench.api.df_store import DFStore

# Create a DFStore object
df_store = DFStore()


def test_repr():
    print("DFStore Object...")
    print(df_store)


def test_summary():
    print("Summary of DataFrames...")
    print(df_store.summary())


def test_details():
    print("Details of DataFrames...")
    print(df_store.details())


def test_upsert_dataframe():
    my_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_store.upsert("/tests/test_data", my_df)


def test_upsert_series():
    series = pd.Series([1, 2, 3, 4], name="Series")
    df_store.upsert("/tests/test_series", series)


def test_get_dataframe():
    return_value = df_store.get("/tests/test_data")
    print(f"Getting data 'test_data':\n{return_value}")


def test_get_series():
    return_value = df_store.get("/tests/test_series")
    print(f"Getting data 'test_series':\n{return_value}")


def test_string_types():
    # Create a DataFrame with int64 values.
    df = pd.DataFrame({"Id": ["A", "B", "C"], "Name": ["Alice", "Bob", "Charlie"]})
    df_store.upsert("/tests/test_string_types", df)
    df_store.delete("/tests/test_string_types")


def test_type_conflation_issues():
    # Create a DataFrame with int64 values.
    df = pd.DataFrame({"Ver": [123, 456, 789]}, dtype="int64")

    # Sneak in a "-" into one value (now column type becomes 'object').
    df.loc[1, "Ver"] = "-"

    df_store.upsert("/tests/test_type_conflation", df)
    df_store.delete("/tests/test_type_conflation")


def test_deletion():
    df_store.delete("/tests/test_data")
    df_store.delete("/tests/test_series")


if __name__ == "__main__":

    # Run the tests
    test_repr()
    test_summary()
    test_details()
    test_upsert_dataframe()
    test_upsert_series()
    test_get_dataframe()
    test_get_series()
    test_repr()
    test_string_types()
    test_type_conflation_issues()
    test_deletion()
