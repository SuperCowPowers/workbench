"""Tests for categorical feature support in FeatureSpaceProximity."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("sklearn") is None,
    reason="pandas and scikit-learn are required for FeatureSpaceProximity tests",
)


def _test_deps():
    import pandas as pd

    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

    return pd, FeatureSpaceProximity


def test_categorical_feature_affects_neighbors():
    pd, FeatureSpaceProximity = _test_deps()
    df = pd.DataFrame(
        {
            "id": ["red-near", "blue-near", "red-far"],
            "size": [1.0, 1.0, 10.0],
            "color": ["red", "blue", "red"],
            "target": [0, 1, 2],
        }
    )

    prox = FeatureSpaceProximity(df, id_column="id", features=["size", "color"], target="target")
    neighbors = prox.neighbors_from_query_df(pd.DataFrame({"size": [1.0], "color": ["blue"]}), n_neighbors=1)

    assert {"color_blue", "color_red"}.issubset(prox._encoded_features)
    assert neighbors.loc[0, "neighbor_id"] == "blue-near"
    assert neighbors.loc[0, "distance"] == 0.0


def test_unseen_query_category_aligns_to_training_columns():
    pd, FeatureSpaceProximity = _test_deps()
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "size": [1.0, 2.0],
            "color": ["red", "blue"],
        }
    )

    prox = FeatureSpaceProximity(df, id_column="id", features=["size", "color"])
    encoded = prox._encode_feature_frame(pd.DataFrame({"size": [1.0], "color": ["green"]}))

    assert encoded.columns.tolist() == prox._encoded_features
    assert "color_green" not in encoded.columns
    assert encoded.filter(like="color_").sum(axis=1).iloc[0] == 0.0
