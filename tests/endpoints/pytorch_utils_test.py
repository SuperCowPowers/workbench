"""Fast unit tests for ``workbench.endpoints.pytorch_utils.prepare_data``.

Focused on categorical handling: ``category_mappings`` is the list-form
``{col: [category, ...]}`` that ``convert_categorical_types`` produces and
``category_mappings.json`` serializes.
"""

import pandas as pd
import pytest

pytest.importorskip("torch")

from workbench.endpoints.inference import convert_categorical_types  # noqa: E402
from workbench.endpoints.pytorch_utils import prepare_data  # noqa: E402


def test_prepare_data_maps_categories_by_list_position():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "color": ["red", "blue", "green"], "y": [0.1, 0.2, 0.3]})
    _, x_cat, _, _, _ = prepare_data(df, ["x1"], ["color"], "y", {"color": ["red", "blue", "green"]})
    assert x_cat.squeeze(1).tolist() == [0, 1, 2]


def test_prepare_data_maps_unknown_categories_to_zero():
    df = pd.DataFrame({"x1": [1.0, 2.0], "color": ["blue", "purple"]})
    _, x_cat, _, _, _ = prepare_data(df, ["x1"], ["color"], category_mappings={"color": ["red", "blue"]})
    assert x_cat.squeeze(1).tolist() == [1, 0]


def test_prepare_data_indices_match_categorical_codes():
    """Training and inference frames aligned by ``convert_categorical_types`` must get
    indices identical to the pandas Categorical codes the embeddings were trained on."""
    train_df = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "color": ["red", "blue", "green", "blue"], "y": [1.0] * 4})
    train_df, mappings = convert_categorical_types(train_df, ["x1", "color"])

    _, x_cat, _, mappings_out, scaler = prepare_data(train_df, ["x1"], ["color"], "y", mappings)
    assert mappings_out is mappings
    assert x_cat.squeeze(1).tolist() == train_df["color"].cat.codes.tolist()

    inf_df = pd.DataFrame({"x1": [5.0, 6.0], "color": ["green", "red"]})
    inf_df, _ = convert_categorical_types(inf_df, ["x1", "color"], mappings)
    _, x_cat_inf, _, _, _ = prepare_data(inf_df, ["x1"], ["color"], category_mappings=mappings, scaler=scaler)
    assert x_cat_inf.squeeze(1).tolist() == [mappings["color"].index("green"), mappings["color"].index("red")]


def test_prepare_data_builds_list_form_mappings_when_none_given():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "color": ["red", "blue", "red"], "y": [0.1, 0.2, 0.3]})
    _, x_cat, _, mappings, _ = prepare_data(df, ["x1"], ["color"], "y")
    assert mappings == {"color": ["red", "blue"]}
    assert x_cat.squeeze(1).tolist() == [0, 1, 0]


def test_prepare_data_no_categoricals():
    df = pd.DataFrame({"x1": [1.0, 2.0], "y": [0.1, 0.2]})
    x_cont, x_cat, y, _, _ = prepare_data(df, ["x1"], [], "y")
    assert x_cat is None
    assert x_cont.shape == (2, 1) and y.shape == (2, 1)
