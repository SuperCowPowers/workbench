"""Tests for PandasToFeatures constructor input options."""

import pandas as pd
import pytest

from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.core.transforms import transform as transform_base


class MockConfigManager:
    """Minimal ConfigManager for constructor-only transform tests."""

    def config_okay(self):
        return True

    def get_config(self, name):
        return "workbench-test-bucket"


class MockAWSSession:
    """Minimal AWS session for constructor-only transform tests."""

    def get_workbench_execution_role_arn(self):
        return "arn:aws:iam::123456789012:role/workbench-test-role"


class MockAWSAccountClamp:
    """Minimal AWSAccountClamp for constructor-only transform tests."""

    def __init__(self):
        self.aws_session = MockAWSSession()
        self.boto3_session = object()

    def sagemaker_session(self):
        return object()

    def sagemaker_client(self):
        return object()


def test(monkeypatch):
    """PandasToFeatures accepts input setup in the constructor."""
    monkeypatch.setattr(transform_base, "ConfigManager", MockConfigManager)
    monkeypatch.setattr(transform_base, "AWSAccountClamp", MockAWSAccountClamp)

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "species": ["cat", "dog"],
            "score": [1.5, 2.0],
            "training": [True, False],
        }
    )

    transform = PandasToFeatures(
        "constructor_features",
        input_df=df,
        id_column="id",
        tags=["test", "constructor"],
        one_hot_columns=["species"],
    )

    assert transform.output_tags == "test::constructor"
    assert transform.id_column == "id"
    assert "event_time" in transform.output_df.columns
    assert "training" not in transform.output_df.columns
    assert "species_cat" in transform.output_df.columns
    assert "species_dog" in transform.output_df.columns
    assert "event_time" not in df.columns

    with pytest.raises(ValueError, match="input_df is required"):
        PandasToFeatures("missing_input", id_column="id")
