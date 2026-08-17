"""Tests for the publish cascade and version drift (no AWS required)

AWS existence checks and the publish steps are stubbed, so these cover the cascade
ordering, the confirm gate, and the drift comparison without touching AWS.
"""

import sys
import types

import pandas as pd
import pytest

from workbench.local import LocalDataSource
from workbench.utils import version_drift
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
def chain():
    """A local DataSource -> FeatureSet chain"""
    df = pd.DataFrame({"id": [1, 2, 3], "x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]})
    ds = LocalDataSource(df, name="pub_data")
    fs = ds.to_features("pub_features", id_column="id")
    return ds, fs


@pytest.fixture
def stub_aws(monkeypatch):
    """Stub the AWS side: nothing exists, and publishing records the call"""
    published = []

    def fake_exists(self):
        return self.name in {a.name for a in published}

    def fake_publish(self, **kwargs):
        published.append(self)
        return f"aws:{self.name}"

    for cls_name in ("local_data_source.LocalDataSource", "local_feature_set.LocalFeatureSet"):
        module, cls = cls_name.split(".")
        target = getattr(__import__(f"workbench.local.{module}", fromlist=[cls]), cls)
        monkeypatch.setattr(target, "aws_exists", fake_exists)
        monkeypatch.setattr(target, "_publish_self", fake_publish)
    return published


class TestPublishPlan:
    def test_plan_walks_lineage_oldest_first(self, chain, stub_aws):
        _, fs = chain
        plan = fs.publish_plan()

        assert [step["name"] for step in plan] == ["pub_data", "pub_features"]
        assert [step["type"] for step in plan] == ["data_source", "feature_set"]
        assert all(step["action"] == "create" for step in plan)

    def test_plan_marks_existing_artifacts(self, chain, stub_aws, monkeypatch):
        ds, fs = chain
        monkeypatch.setattr(type(ds), "aws_exists", lambda self: self.artifact_type == "data_source")

        plan = fs.publish_plan()
        assert plan[0]["action"] == "exists"
        assert plan[1]["action"] == "create"

    def test_root_plan_is_just_itself(self, chain, stub_aws):
        ds, _ = chain
        assert [step["name"] for step in ds.publish_plan()] == ["pub_data"]

    def test_orphaned_artifact_has_no_parent(self, chain, stub_aws):
        """A FeatureSet whose DataSource was deleted publishes alone"""
        ds, fs = chain
        ds.delete()
        assert [step["name"] for step in fs.publish_plan()] == ["pub_features"]


class TestConfirmGate:
    def test_without_confirm_nothing_is_published(self, chain, stub_aws):
        _, fs = chain
        result = fs.publish()

        assert stub_aws == []
        assert [step["name"] for step in result] == ["pub_data", "pub_features"]

    def test_with_confirm_publishes_whole_chain_in_order(self, chain, stub_aws):
        _, fs = chain
        result = fs.publish(confirm=True)

        assert [a.name for a in stub_aws] == ["pub_data", "pub_features"]
        assert result == "aws:pub_features"

    def test_existing_artifacts_are_skipped(self, chain, stub_aws, monkeypatch):
        ds, fs = chain
        monkeypatch.setattr(type(ds), "aws_exists", lambda self: self.artifact_type == "data_source")

        fs.publish(confirm=True)
        assert [a.name for a in stub_aws] == ["pub_features"]

    def test_returns_self_not_an_ancestor(self, chain, stub_aws, monkeypatch):
        """Publishing a chain whose tail already exists must still return the tail"""
        ds, fs = chain
        monkeypatch.setattr(type(fs), "aws_exists", lambda self: self.artifact_type == "feature_set")
        monkeypatch.setattr(type(fs), "_aws_artifact", lambda self: f"aws:{self.name}")

        result = fs.publish(confirm=True)

        # The DataSource gets created, but the caller asked for the FeatureSet
        assert [a.name for a in stub_aws] == ["pub_data"]
        assert result == "aws:pub_features"


class TestRowRolesSurvivePublish:
    """publish() retrains in AWS, so every role the local run used has to be replayed.
    A dropped role is a model that silently differs from the one that was validated."""

    def test_weights_as_pairs_round_trip(self):
        from workbench.local.local_model import LocalModel

        pairs = LocalModel._weights_as_pairs({1: 2.5, 2: 0.5})
        assert dict(pairs) == {1: 2.5, 2: 0.5}

        # Integer ids must stay integers: JSON object keys would stringify them and
        # then fail to join against the FeatureSet id column
        assert all(isinstance(key, int) for key, _ in pairs)

    def test_weights_from_dataframe(self):
        from workbench.local.local_model import LocalModel

        weights_df = pd.DataFrame({"id": [1, 2], "sample_weight": [3.0, 0.25]})
        assert dict(LocalModel._weights_as_pairs(weights_df)) == {1: 3.0, 2: 0.25}

    def test_no_weights_is_none(self):
        from workbench.local.local_model import LocalModel

        assert LocalModel._weights_as_pairs(None) is None
        assert LocalModel._weights_as_pairs({}) is None
        assert LocalModel._weights_as_pairs(pd.DataFrame()) is None

    def test_publish_replays_all_three_roles(self, monkeypatch):
        """The AWS retrain must receive the weights, not just the id lists"""
        from workbench.local.local_model import LocalModel

        model = LocalModel("role-model")
        model.upsert_workbench_meta(
            {
                "model_type": "regressor",
                "model_framework": "xgboost",
                "workbench_model_target": "y",
                "workbench_model_features": ["a"],
                "workbench_input": "some_features",
                "sample_weights": [[1, 2.5]],
                "validation_ids": [3],
                "exclude_ids": [4],
            }
        )

        captured = {}

        class FakeFeatureSet:
            def __init__(self, name):
                pass

            def to_model(self, **kwargs):
                captured.update(kwargs)
                return "aws-model"

        # Stand in for workbench.api entirely: importing it builds an AWS session
        fake_api = types.ModuleType("workbench.api")
        fake_api.FeatureSet = FakeFeatureSet
        monkeypatch.setitem(sys.modules, "workbench.api", fake_api)
        monkeypatch.setattr(LocalModel, "version_drift", lambda self: "")

        assert model._publish_self() == "aws-model"
        assert captured["sample_weights"] == {1: 2.5}
        assert captured["validation_ids"] == [3]
        assert captured["exclude_ids"] == [4]


class TestVersionDrift:
    def test_build_suffixes_compare_equal(self):
        """A local CPU torch build matches the image's CUDA build"""
        assert version_drift.base_version("2.13.0+cu130") == version_drift.base_version("2.13.0")

    def test_lock_parsing(self, tmp_path):
        lock = tmp_path / "requirements.lock"
        lock.write_text("# comment\nxgboost==3.2.0\ntorch==2.13.0+cu130\n\n-e .\n")

        pins = version_drift.parse_lock(lock)
        assert pins == {"xgboost": "3.2.0", "torch": "2.13.0+cu130"}

    def test_missing_lock_reports_no_drift(self, tmp_path):
        assert version_drift.parse_lock(tmp_path / "nope.lock") == {}

    def test_framework_picks_the_right_image(self):
        assert "pytorch_chem" in str(version_drift.image_lock_path("chemprop"))
        assert "pytorch_chem" in str(version_drift.image_lock_path("pytorch"))
        assert "base" in str(version_drift.image_lock_path("xgboost"))

    def test_drift_detected(self, monkeypatch):
        monkeypatch.setattr(version_drift, "parse_lock", lambda path: {"xgboost": "0.0.1"})
        drift = version_drift.check_drift("xgboost", packages=["xgboost"])

        assert len(drift) == 1
        assert drift[0]["image"] == "0.0.1"
        assert drift[0]["package"] == "xgboost"

    def test_packages_absent_locally_are_skipped(self, monkeypatch):
        monkeypatch.setattr(version_drift, "parse_lock", lambda path: {"not-a-real-package": "1.0"})
        assert version_drift.check_drift("xgboost", packages=["not-a-real-package"]) == []


class TestWorkbenchDrift:
    """The model scripts import workbench from inside the image, so a version gap there
    is an ImportError at training time rather than a numerical difference."""

    def test_image_version_read_from_dockerfile(self):
        assert version_drift.image_workbench_version("xgboost")
        assert version_drift.image_workbench_version("chemprop")

    def test_drift_reported_when_versions_differ(self, monkeypatch):
        monkeypatch.setattr(version_drift, "image_workbench_version", lambda fw, stage="training": "0.0.1")
        drift = version_drift.check_workbench_drift("xgboost")
        assert drift["package"] == "workbench"
        assert drift["image"] == "0.0.1"

    def test_no_drift_when_versions_match(self, monkeypatch):
        from importlib.metadata import version as pkg_version

        monkeypatch.setattr(
            version_drift, "image_workbench_version", lambda fw, stage="training": pkg_version("workbench")
        )
        assert version_drift.check_workbench_drift("xgboost") == {}

    def test_summary_explains_the_import_failure(self, monkeypatch):
        monkeypatch.setattr(version_drift, "image_workbench_version", lambda fw, stage="training": "0.0.1")
        monkeypatch.setattr(version_drift, "check_drift", lambda fw: [])
        summary = version_drift.drift_summary("xgboost")
        assert "workbench: local" in summary
        assert "fails at import" in summary

    def test_images_dir_found_from_a_source_checkout(self):
        assert version_drift.images_dir() is not None

    def test_unavailable_images_say_so_rather_than_reporting_no_drift(self, monkeypatch):
        """Silence would read as 'versions match', which is the opposite of what we know"""
        monkeypatch.setattr(version_drift, "images_dir", lambda: None)

        assert version_drift.image_lock_path("xgboost") is None
        assert version_drift.parse_lock(None) == {}
        assert version_drift.image_workbench_version("xgboost") == ""
        assert "Could not verify" in version_drift.drift_summary("xgboost")


class TestSaveOutput:
    """S3 behavior must stay exactly what it was before local writes were added"""

    def test_local_dataframe_and_object(self, tmp_path):
        from workbench.endpoints.inference import save_output

        save_output(pd.DataFrame({"a": [1, 2]}), str(tmp_path / "sub" / "preds.csv"))
        save_output({"lr": 0.1}, str(tmp_path / "params.json"))

        assert (tmp_path / "sub" / "preds.csv").read_text().strip() == "a\n1\n2"
        assert (tmp_path / "params.json").read_text() == '{"lr": 0.1}'

    def test_bare_filename_has_no_directory_to_create(self, tmp_path, monkeypatch):
        """os.path.dirname is empty for a bare name, which makedirs rejects"""
        from workbench.endpoints.inference import save_output

        monkeypatch.chdir(tmp_path)
        save_output({"a": 1}, "bare.json")
        assert (tmp_path / "bare.json").read_text() == '{"a": 1}'

    def test_s3_dataframe_streams_through_awswrangler(self):
        """put_object would hold the whole CSV in memory and caps at 5GB"""
        import sys
        import types
        from unittest import mock
        from workbench.endpoints.inference import save_output

        fake_wr = types.ModuleType("awswrangler")
        fake_wr.s3 = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"awswrangler": fake_wr}):
            with mock.patch("boto3.client") as boto:
                save_output(pd.DataFrame({"a": [1]}), "s3://bucket/key/preds.csv")

        fake_wr.s3.to_csv.assert_called_once()
        assert fake_wr.s3.to_csv.call_args.kwargs == {"index": False}
        assert not boto.return_value.put_object.called

    def test_s3_object_uses_put_object(self):
        from unittest import mock
        from workbench.endpoints.inference import save_output

        with mock.patch("boto3.client") as boto:
            save_output({"x": 1}, "s3://bucket/key/params.json")

        kwargs = boto.return_value.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "bucket"
        assert kwargs["Key"] == "key/params.json"
        assert kwargs["Body"] == '{"x": 1}'
