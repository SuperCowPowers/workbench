"""Tests for the Workbench Views functionality"""

import pytest
import logging
import pandas as pd

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
    """Test set_sample_weights with dict, DataFrame, get_sample_weights, and add_filter"""
    fs = FeatureSet("test_features")
    total_rows = fs.num_rows()

    # Get all IDs so we can pick a few to zero out
    df = fs.pull_dataframe()
    id_col = fs.id_column
    ids = df[id_col].tolist()

    # --- Test 1: Dict input ---
    weights = {ids[0]: 0.0, ids[1]: 0.0, ids[2]: 0.0, ids[3]: 0.5}
    fs.set_sample_weights(weights)

    tv_df = fs.view("training").pull_dataframe()
    assert "sample_weight" in tv_df.columns
    assert len(tv_df) == total_rows - 3
    assert ids[0] not in tv_df[id_col].values
    row = tv_df[tv_df[id_col] == ids[3]]
    assert row["sample_weight"].iloc[0] == 0.5
    normal_row = tv_df[tv_df[id_col] == ids[4]]
    assert normal_row["sample_weight"].iloc[0] == 1.0

    # --- Test 2: get_sample_weights returns DataFrame ---
    read_weights = fs.get_sample_weights()
    assert isinstance(read_weights, pd.DataFrame)
    assert list(read_weights.columns) == [id_col, "sample_weight"]
    assert len(read_weights) == 4
    w_dict = read_weights.set_index(id_col)["sample_weight"].to_dict()
    assert w_dict[ids[0]] == 0.0
    assert w_dict[ids[3]] == 0.5

    # --- Test 3: DataFrame input ---
    weights_df = pd.DataFrame({id_col: [ids[0], ids[1]], "sample_weight": [0.0, 0.3]})
    fs.set_sample_weights(weights_df)

    tv_df = fs.view("training").pull_dataframe()
    assert len(tv_df) == total_rows - 1  # only ids[0] excluded (weight 0.0)
    assert ids[0] not in tv_df[id_col].values
    row = tv_df[tv_df[id_col] == ids[1]]
    assert row["sample_weight"].iloc[0] == 0.3

    # --- Test 4: add_filter ---
    # Start fresh: set ids[3] to 0.5
    fs.set_sample_weights({ids[3]: 0.5})
    # Now additively filter out ids[4] and ids[5]
    fs.add_filter([ids[4], ids[5]])

    tv_df = fs.view("training").pull_dataframe()
    assert len(tv_df) == total_rows - 2  # ids[4] and ids[5] excluded
    assert ids[4] not in tv_df[id_col].values
    assert ids[5] not in tv_df[id_col].values
    # Verify original weight for ids[3] is preserved
    row = tv_df[tv_df[id_col] == ids[3]]
    assert row["sample_weight"].iloc[0] == 0.5

    # --- Cleanup: Reset to standard training view ---
    fs.set_sample_weights({})
    tv_df = fs.view("training").pull_dataframe()
    assert len(tv_df) == total_rows
    # Verify weights table is cleaned up
    read_weights = fs.get_sample_weights()
    assert read_weights.empty
    print(f"set_sample_weights test passed: dict, DataFrame, get_sample_weights, add_filter, reset all verified")


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
