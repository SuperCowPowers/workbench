"""Tests for the Workbench Views functionality"""

import pytest
import logging

# Workbench Imports
from workbench.api import DataSource, FeatureSet
from workbench.core.views import View

# Show debug calls
logging.getLogger("workbench").setLevel(logging.DEBUG)


def test_display_view_ds():
    # Grab the Display View for a DataSource
    data_source = DataSource("abalone_data")
    display_view = View(data_source, "display")
    df = display_view.pull_dataframe()
    print(df)


def test_display_view_fs():
    # Grab the display View for a FeatureSet
    fs = FeatureSet("test_features")
    display_view = View(fs, "display")
    df = display_view.pull_dataframe(limit=5)
    print(df)


@pytest.mark.long
def test_set_computation_view_columns():
    # Create a new computation View for a DataSource
    ds = DataSource("test_data")
    computation_columns = ds.columns
    ds.set_computation_columns(computation_columns)
    computation_view = ds.view("computation")
    df = computation_view.pull_dataframe(limit=5)
    print(df)

    # Create a new display View for a FeatureSet
    fs = FeatureSet("test_features")
    fs.set_computation_columns(computation_columns)
    computation_view = fs.view("display")
    df = computation_view.pull_dataframe(limit=5)
    print(df)


def test_delete_display_view():
    # Delete the display view
    fs = FeatureSet("test_features")
    display_view = View(fs, "display")
    display_view.delete()
    fs.view("display")


def test_computation_view():
    # Grab a Computation View for a FeatureSet
    fs = FeatureSet("test_features")
    computation_view = View(fs, "computation")
    df = computation_view.pull_dataframe()
    print(df)


def test_view_on_non_existent_data():
    # Create a View for the Non-Existing DataSource
    data_source = DataSource("non_existent_data")
    assert View(data_source, "display").exists() is False
    assert View(data_source, "computation").exists() is False


if __name__ == "__main__":

    # Run the tests
    test_display_view_ds()
    test_display_view_fs()
    test_set_computation_view_columns()
    test_delete_display_view()
    test_computation_view()
    test_view_on_non_existent_data()
