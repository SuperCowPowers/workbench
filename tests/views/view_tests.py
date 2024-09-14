"""Tests for the SageWorks Views functionality"""

import pytest
import logging

# SageWorks Imports
from sageworks.api import DataSource, FeatureSet
from sageworks.core.views import View

# Show debug calls
logging.getLogger("sageworks").setLevel(logging.DEBUG)


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
    df = display_view.pull_dataframe(head=True)
    print(df)


@pytest.mark.long
def test_set_computation_view_columns():
    # Create a new computation View for a DataSource
    ds = DataSource("test_data")
    computation_columns = ds.columns
    ds.set_computation_columns(computation_columns)
    computation_view = ds.view("computation")
    df = computation_view.pull_dataframe(head=True)
    print(df)

    # Create a new display View for a FeatureSet
    fs = FeatureSet("test_features")
    fs.set_computation_columns(computation_columns)
    computation_view = fs.view("display")
    df = computation_view.pull_dataframe(head=True)
    print(df)


def test_training_view():
    # Grab the Training View for a FeatureSet
    fs = FeatureSet("test_features")
    training_view = View(fs, "training")
    df = training_view.pull_dataframe(head=True)
    print(df)


def test_training_holdouts():
    # Test setting the training holdouts (which creates a new training view)
    fs = FeatureSet("test_features")
    df = fs.get_training_data()
    print(f"Before setting holdouts: {df['training'].value_counts()}")
    fs.set_training_holdouts("id", [1, 2, 3])
    training_view = fs.view("training")
    df = training_view.pull_dataframe()
    print(f"After setting holdouts: {df['training'].value_counts()}")


def test_delete_display_view():
    # Delete the display view
    fs = FeatureSet("test_features")
    display_view = View(fs, "display")
    display_view.delete()
    fs.view("display")


def test_delete_training_view():
    # Delete the training view
    fs = FeatureSet("test_features")
    training_view = View(fs, "training")
    training_view.delete()
    fs.view("training")


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
    assert View(data_source, "training").exists() is False


if __name__ == "__main__":

    # Run the tests
    test_display_view_ds()
    test_display_view_fs()
    test_set_computation_view_columns()
    test_training_view()
    test_training_holdouts()
    test_delete_display_view()
    test_delete_training_view()
    test_computation_view()
    test_view_on_non_existent_data()
