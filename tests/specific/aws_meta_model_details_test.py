# isort: skip_file
"""Unit coverage for lightweight SageMaker model metadata summaries."""

import importlib.util
import os
import sys
import types
from datetime import datetime, timezone

os.environ["WORKBENCH_SKIP_LOGGING"] = "true"


def _install_optional_dependency_stubs():
    """Keep this focused unit test runnable without the full AWS dependency stack."""

    if importlib.util.find_spec("pandas") is None:
        pandas = types.ModuleType("pandas")

        class DataFrame:
            def __init__(self, rows=None):
                self.rows = list(rows or [])

            @property
            def empty(self):
                return not self.rows

            @property
            def loc(self):
                frame = self

                class Loc:
                    def __getitem__(self, key):
                        row_index, column = key
                        return frame.rows[row_index][column]

                return Loc()

            def convert_dtypes(self):
                return self

            def sort_values(self, by, ascending=True, inplace=False):
                rows = sorted(self.rows, key=lambda row: row[by], reverse=not ascending)
                if inplace:
                    self.rows = rows
                    return None
                return DataFrame(rows)

        pandas.DataFrame = DataFrame
        sys.modules["pandas"] = pandas

    if importlib.util.find_spec("awswrangler") is None:
        sys.modules["awswrangler"] = types.ModuleType("awswrangler")

    if importlib.util.find_spec("boto3") is not None and importlib.util.find_spec("botocore") is not None:
        return

    account_clamp = types.ModuleType("workbench.core.cloud_platform.aws.aws_account_clamp")

    class AWSAccountClamp:
        pass

    account_clamp.AWSAccountClamp = AWSAccountClamp
    sys.modules["workbench.core.cloud_platform.aws.aws_account_clamp"] = account_clamp

    config_manager = types.ModuleType("workbench.utils.config_manager")

    class ConfigManager:
        pass

    config_manager.ConfigManager = ConfigManager
    sys.modules["workbench.utils.config_manager"] = config_manager

    aws_utils = types.ModuleType("workbench.utils.aws_utils")

    def identity_decorator(func=None, **_kwargs):
        if func is None:
            return lambda inner: inner
        return func

    aws_utils.not_found_returns_none = identity_decorator
    aws_utils.aws_throttle = identity_decorator
    aws_utils.aws_tags_to_dict = lambda tags: {tag["Key"]: tag["Value"] for tag in tags}
    sys.modules["workbench.utils.aws_utils"] = aws_utils


_install_optional_dependency_stubs()

from workbench.core.cloud_platform.aws.aws_meta import AWSMeta  # noqa: E402


class FakePaginator:
    def __init__(self, groups):
        self.groups = groups

    def paginate(self):
        return [{"ModelPackageGroupSummaryList": self.groups}]


class FakeSageMakerClient:
    def __init__(self):
        self.calls = []
        self.group = {
            "ModelPackageGroupName": "demo-model",
            "ModelPackageGroupArn": "arn:aws:sagemaker:us-east-1:123:model-package-group/demo-model",
            "CreationTime": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "ModelPackageGroupDescription": "demo description",
        }
        self.latest_package = {
            "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:model-package/demo-model/7",
            "ModelPackageStatus": "Completed",
            "ModelPackageVersion": 7,
        }

    def get_paginator(self, operation):
        self.calls.append(("get_paginator", operation))
        return FakePaginator([self.group])

    def describe_model_package_group(self, **kwargs):
        self.calls.append(("describe_model_package_group", kwargs))
        return self.group

    def list_model_packages(self, **kwargs):
        self.calls.append(("list_model_packages", kwargs))
        return {"ModelPackageSummaryList": [self.latest_package]}

    def describe_model_package(self, **kwargs):
        self.calls.append(("describe_model_package", kwargs))
        raise AssertionError("model summaries should not call describe_model_package")


def make_meta(client):
    meta = AWSMeta.__new__(AWSMeta)
    meta.sm_client = client
    meta.boto3_session = types.SimpleNamespace(region_name="us-east-1")
    meta.account_clamp = types.SimpleNamespace(region="us-east-1")
    meta.get_aws_tags = lambda arn: {
        "workbench_owner": "owner",
        "workbench_model_type": "classifier",
        "workbench_model_framework": "xgboost",
        "workbench_input": "features",
        "workbench_tags": "tag-a",
        "workbench_health_tags": "healthy",
    }
    return meta


def test_models_details_reuses_group_and_package_summaries():
    client = FakeSageMakerClient()
    meta = make_meta(client)

    df = meta.models(details=True)

    assert df.loc[0, "Model Group"] == "demo-model"
    assert df.loc[0, "Ver"] == 7
    assert df.loc[0, "Status"] == "Completed"
    assert df.loc[0, "Owner"] == "owner"
    assert "describe_model_package_group" not in [call[0] for call in client.calls]
    assert "describe_model_package" not in [call[0] for call in client.calls]


def test_model_detail_row_uses_latest_package_summary_for_version_and_status():
    client = FakeSageMakerClient()
    meta = make_meta(client)

    row = meta._model_detail_row("demo-model")

    assert row["Ver"] == 7
    assert row["Status"] == "Completed"
    assert row["Description"] == "demo description"
    assert "describe_model_package" not in [call[0] for call in client.calls]
