"""Tests for Athena query output path configuration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeParameterStore:
    value = None

    def get(self, key):
        return self.value


def load_athena_utils(monkeypatch):
    """Load athena_utils with AWS/dataframe dependencies stubbed."""
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.session = types.SimpleNamespace(Session=object)
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = object
    fake_wr = types.ModuleType("awswrangler")
    fake_wr.s3 = types.SimpleNamespace(delete_objects=lambda *args, **kwargs: None)

    perf_module = types.ModuleType("workbench.utils.performance_utils")
    perf_module.performance = lambda func: func
    boto_session_module = types.ModuleType("workbench.core.cloud_platform.aws.boto_session")
    boto_session_module.get_boto3_session = lambda: None
    parameter_store_module = types.ModuleType("workbench.core.parameter_store_core")
    parameter_store_module.ParameterStoreCore = FakeParameterStore

    for name, module in {
        "boto3": fake_boto3,
        "pandas": fake_pandas,
        "awswrangler": fake_wr,
        "workbench.utils.performance_utils": perf_module,
        "workbench.core.cloud_platform.aws.boto_session": boto_session_module,
        "workbench.core.parameter_store_core": parameter_store_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[2] / "src" / "workbench" / "utils" / "athena_utils.py"
    spec = importlib.util.spec_from_file_location("athena_utils_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_athena_output_s3_path_uses_explicit_bucket(monkeypatch):
    athena_utils = load_athena_utils(monkeypatch)

    assert athena_utils.athena_output_s3_path("test-bucket") == "s3://test-bucket/temp/athena_output"


def test_athena_output_s3_path_falls_back_to_environment(monkeypatch):
    FakeParameterStore.value = None
    monkeypatch.setenv("WORKBENCH_BUCKET", "env-bucket")
    athena_utils = load_athena_utils(monkeypatch)

    assert athena_utils.athena_output_s3_path() == "s3://env-bucket/temp/athena_output"


def test_athena_output_s3_path_requires_bucket(monkeypatch):
    FakeParameterStore.value = None
    monkeypatch.delenv("WORKBENCH_BUCKET", raising=False)
    athena_utils = load_athena_utils(monkeypatch)

    with pytest.raises(ValueError, match="WORKBENCH_BUCKET"):
        athena_utils.athena_output_s3_path()
