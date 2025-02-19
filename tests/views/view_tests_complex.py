"""Tests for the Workbench Views functionality"""

import numpy as np
import logging

# Workbench Imports
from workbench.api import DataSource, FeatureSet
from workbench.core.views import PandasToView

# Show debug calls
logging.getLogger("workbench").setLevel(logging.DEBUG)


def test_pandas_to_ds_view():
    # Create a new PandasToView for a DataSource
    ds = DataSource("test_data")
    df = ds.pull_dataframe()
    df["random1"] = np.random.rand(len(df))
    df["random2"] = np.random.rand(len(df))
    PandasToView.create("test_view", ds, df, "id")
    test_view = ds.view("test_view")
    df = test_view.pull_dataframe(limit=5)
    print(df)
    test_view.delete()


def test_pandas_to_fs_view():
    # Create a new PandasToView for a FeatureSet
    fs = FeatureSet("test_features")
    df = fs.pull_dataframe()
    df["random1"] = np.random.rand(len(df))
    df["random2"] = np.random.rand(len(df))
    PandasToView.create("test_view", fs, df, "id")
    test_view = fs.view("test_view")
    df = test_view.pull_dataframe(limit=5)
    print(df)
    test_view.delete()


if __name__ == "__main__":

    # Run the tests
    test_pandas_to_ds_view()
    test_pandas_to_fs_view()
