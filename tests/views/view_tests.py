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


def test_training_view():
    # Grab the Training View for a FeatureSet
    fs = FeatureSet("test_features")
    training_view = View(fs, "training")
    df = training_view.pull_dataframe(limit=5)
    print(df)


def test_training_holdouts():
    # Test setting the training holdouts (which creates a new training view)
    fs = FeatureSet("test_features")
    df = fs.get_training_data()
    print(f"Before setting holdouts: {df['training'].value_counts()}")
    fs.set_training_holdouts([1, 2, 3])
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


def test_set_sample_weights():
    """Test set_sample_weights with a small dict (uses supplemental table)"""
    fs = FeatureSet("test_features")
    total_rows = fs.num_rows()

    # Get all IDs so we can pick a few to zero out
    df = fs.pull_dataframe()
    id_col = fs.id_column
    ids = df[id_col].tolist()

    # Set weights: exclude first 3 IDs, downweight the 4th
    weights = {ids[0]: 0.0, ids[1]: 0.0, ids[2]: 0.0, ids[3]: 0.5}
    fs.set_sample_weights(weights)

    # Pull the training view and verify
    tv_df = fs.view("training").pull_dataframe()
    assert "sample_weight" in tv_df.columns
    assert len(tv_df) == total_rows - 3  # 3 zero-weight rows excluded
    assert ids[0] not in tv_df[id_col].values
    assert ids[1] not in tv_df[id_col].values
    assert ids[2] not in tv_df[id_col].values

    # Check the downweighted row
    row = tv_df[tv_df[id_col] == ids[3]]
    assert len(row) == 1
    assert row["sample_weight"].iloc[0] == 0.5

    # Check a normal row has default weight 1.0
    normal_row = tv_df[tv_df[id_col] == ids[4]]
    assert normal_row["sample_weight"].iloc[0] == 1.0

    # Verify get_sample_weights reads back what we set
    read_weights = fs.get_sample_weights()
    assert read_weights[ids[0]] == 0.0
    assert read_weights[ids[1]] == 0.0
    assert read_weights[ids[2]] == 0.0
    assert read_weights[ids[3]] == 0.5
    assert len(read_weights) == 4  # only explicitly set IDs

    # Reset to standard training view
    fs.set_sample_weights({})
    tv_df = fs.view("training").pull_dataframe()
    assert len(tv_df) == total_rows
    print(f"set_sample_weights test passed: {total_rows} rows, 3 excluded, 1 downweighted, then reset")


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
    test_set_sample_weights()
    test_delete_display_view()
    test_delete_training_view()
    test_computation_view()
    test_view_on_non_existent_data()
