"""Tests for the workbench.local artifacts (no AWS required)"""

import numpy as np
import pandas as pd
import pytest

from workbench.local import LocalDataSource, LocalMeta
from workbench.utils.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "Height": [0.1, 0.4, 0.8, 0.6, 0.2],
            "Food": ["steak", "tofu", "steak", "fish", "tofu"],
            "When": pd.to_datetime(["2026-01-01"] * 5),
            "Nullable": pd.array([1, 2, None, 4, 5], dtype="Int64"),
        }
    )


class TestLocalDataSource:
    def test_create_and_reload(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        assert ds.exists()
        assert ds.num_rows() == 5
        assert ds.num_columns() == 5

        # A bare name reloads the existing artifact
        assert LocalDataSource("my_data").num_rows() == 5

    def test_requires_name_for_dataframe(self, sample_df):
        with pytest.raises(ValueError):
            LocalDataSource(sample_df)

    def test_query(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        result = ds.query("select id, Height from my_data where Height > 0.3 order by Height")
        assert list(result["id"]) == [2, 4, 3]

    def test_pull_dataframe_limit(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        assert len(ds.pull_dataframe()) == 5
        assert len(ds.pull_dataframe(limit=2)) == 2

    def test_missing_artifact_returns_empty(self):
        assert LocalDataSource("nope").pull_dataframe().empty


class TestLocalFeatureSet:
    def test_to_features_column_prep(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        fs = ds.to_features("my_features", id_column="id", one_hot_columns=["Food"])

        # Columns are lowercased and one-hot columns expanded
        assert "height" in fs.columns
        assert "Height" not in fs.columns
        assert {"food_steak", "food_tofu", "food_fish"}.issubset(set(fs.columns))
        assert "food" not in fs.columns

    def test_to_features_dtype_coercion(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        fs = ds.to_features("my_features", id_column="id", one_hot_columns=["Food"])
        df = fs.pull_dataframe()

        # Feature Store type set: datetime -> string, Int64 with NAs -> float64, dummies -> int32
        assert df["when"].dtype == "string"
        assert df["nullable"].dtype == np.float64
        assert df["food_steak"].dtype == np.int32

    def test_mixed_case_id_column(self, sample_df):
        """Column names lowercase, and the id reference follows them down"""
        ds = LocalDataSource(sample_df.rename(columns={"id": "ID"}), name="my_data")
        fs = ds.to_features("my_features", id_column="ID")
        assert fs.id_column == "id"

    def test_auto_id_column(self, sample_df):
        ds = LocalDataSource(sample_df.drop(columns=["id"]), name="my_data")
        fs = ds.to_features("my_features", id_column="auto")
        assert fs.id_column == "auto_id"

    def test_missing_id_column_raises(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        with pytest.raises(ValueError):
            ds.to_features("my_features", id_column="not_a_column")

    def test_lineage_recorded(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        fs = ds.to_features("my_features", id_column="id")
        assert fs.get_input() == "my_data"


class TestTrainingView:
    @pytest.fixture
    def feature_set(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        return ds.to_features("my_features", id_column="id")

    def test_defaults(self, feature_set):
        tv = feature_set.training_view()
        assert len(tv) == 5
        assert (tv["sample_weight"] == 1.0).all()
        assert not tv["validation"].any()
        assert not tv["exclude"].any()

    def test_validation_and_exclude(self, feature_set):
        tv = feature_set.training_view(validation_ids=[3], exclude_ids=[4])

        # Excluded rows are dropped entirely, validation rows are kept and marked
        assert 4 not in set(tv["id"])
        assert len(tv) == 4
        assert tv.loc[tv["id"] == 3, "validation"].item()

    def test_exclude_wins_over_validation(self, feature_set):
        tv = feature_set.training_view(validation_ids=[2], exclude_ids=[2])
        assert 2 not in set(tv["id"])

    def test_sample_weights(self, feature_set):
        tv = feature_set.training_view(sample_weights={1: 2.5})
        assert tv.loc[tv["id"] == 1, "sample_weight"].item() == 2.5
        assert tv.loc[tv["id"] == 2, "sample_weight"].item() == 1.0


class TestLocalMeta:
    def test_listings(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        ds.to_features("my_features", id_column="id")

        meta = LocalMeta()
        assert list(meta.data_sources()["Name"]) == ["my_data"]
        assert list(meta.feature_sets()["Name"]) == ["my_features"]
        assert meta.feature_sets()["Input"].item() == "my_data"

    def test_empty_listings_keep_columns(self):
        models = LocalMeta().models()
        assert models.empty
        assert "Framework" in models.columns

    def test_tags_and_owner(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        ds.add_tag("experimental")
        ds.set_owner("briford")

        row = LocalMeta().data_sources().iloc[0]
        assert "experimental" in row["Tags"]
        assert row["Owner"] == "briford"


class TestLocalArtifactMeta:
    def test_status_and_delete(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        assert ds.get_status() == "ready"
        assert ds.ready()

        ds.set_status("stale")
        assert LocalDataSource("my_data").get_status() == "stale"

        ds.delete()
        assert not ds.exists()

    def test_no_aws_surface(self, sample_df):
        ds = LocalDataSource(sample_df, name="my_data")
        assert not hasattr(ds, "arn")
        assert not hasattr(ds, "aws_url")
        assert "aws_arn" not in ds.summary()
