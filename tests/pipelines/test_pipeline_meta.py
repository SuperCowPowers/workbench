"""Tests for the PipelineMeta class"""

import json
import os
from unittest.mock import patch

import pytest

from workbench.core.pipelines.pipeline_meta import PipelineMeta


class TestPipelineMetaWithEnvVar:
    """Tests for PipelineMeta when PIPELINE_META env var is set."""

    def test_basic_dt_mode(self):
        meta = {
            "mode": "dt",
            "model_name": "my-model-xgb-1-dt",
            "endpoint_name": "my-endpoint-xgb-1-dt",
            "serverless": True,
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "dt"
            assert pm.model_name == "my-model-xgb-1-dt"
            assert pm.endpoint_name == "my-endpoint-xgb-1-dt"
            assert pm.serverless is True

    def test_promote_mode(self):
        meta = {
            "mode": "promote",
            "model_name": "my-model-xgb-1-20260222",
            "endpoint_name": "my-endpoint-xgb-1",
            "serverless": False,
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "promote"
            assert pm.model_name == "my-model-xgb-1-20260222"
            assert pm.endpoint_name == "my-endpoint-xgb-1"
            assert pm.serverless is False

    def test_temporal_split_mode(self):
        meta = {
            "mode": "temporal_split",
            "model_name": "my-model-xgb-1-ts",
            "endpoint_name": "my-endpoint-xgb-1-ts",
            "serverless": True,
            "cutoff_date": "2025-10-17",
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "temporal_split"
            assert pm.get("cutoff_date") == "2025-10-17"

    def test_custom_keys(self):
        meta = {
            "mode": "dt",
            "model_name": "my-model-dt",
            "endpoint_name": "my-endpoint-dt",
            "serverless": True,
            "custom_key": "custom_value",
            "batch_size": 32,
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.get("custom_key") == "custom_value"
            assert pm.get("batch_size") == 32
            assert pm.get("nonexistent", "fallback") == "fallback"

    def test_partial_meta_defaults_mode_and_serverless(self):
        """Partial PIPELINE_META should default mode and serverless only."""
        meta = {"model_name": "my-model", "endpoint_name": "my-endpoint"}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "dt"
            assert pm.serverless is True
            assert pm.model_name == "my-model"
            assert pm.endpoint_name == "my-endpoint"

    def test_get_with_explicit_default(self):
        """get() with an explicit default returns it when key is missing."""
        meta = {"mode": "dt", "model_name": "m", "endpoint_name": "e", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.get("nonexistent", "fallback") == "fallback"
            assert pm.get("nonexistent", None) is None

    def test_get_with_explicit_default_none(self):
        """get() with explicit default=None should return None, not raise."""
        meta = {"mode": "dt", "model_name": "m", "endpoint_name": "e", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.get("nonexistent", None) is None


class TestPipelineMetaFailHard:
    """Tests for fail-hard behavior when env var is missing or keys are missing."""

    def test_no_env_var_raises(self):
        """RuntimeError when PIPELINE_META is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="PIPELINE_META environment variable not set"):
                PipelineMeta()

    def test_invalid_json_raises(self):
        """RuntimeError when PIPELINE_META contains invalid JSON."""
        with patch.dict(os.environ, {"PIPELINE_META": "not-valid-json"}, clear=True):
            with pytest.raises(RuntimeError, match="Failed to parse PIPELINE_META"):
                PipelineMeta()

    def test_get_missing_key_raises(self):
        """RuntimeError when get() is called on a missing key with no default."""
        meta = {"mode": "dt", "model_name": "m", "endpoint_name": "e", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            with pytest.raises(RuntimeError, match="Key 'nonexistent' not found"):
                pm.get("nonexistent")

    def test_dynamic_owner_without_set_owner_defaults_to_test(self):
        """dynamic_owner() returns 'DT' for dt mode when set_owner() hasn't been called."""
        meta = {"mode": "dt", "model_name": "m", "endpoint_name": "e", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.dynamic_owner() == "DT"


class TestPipelineMetaOwner:
    """Tests for set_owner() and dynamic_owner()."""

    def test_dt_mode_owner(self):
        meta = {"mode": "dt", "model_name": "m-dt", "endpoint_name": "e-dt", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("BW")
            assert pm.dynamic_owner() == "DT"

    def test_temporal_split_owner(self):
        meta = {"mode": "ts", "model_name": "m-ts", "endpoint_name": "e-ts", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("BW")
            assert pm.dynamic_owner() == "DT"

    def test_promote_owner(self):
        meta = {"mode": "promote", "model_name": "m-260222", "endpoint_name": "e-1", "serverless": False}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("BW")
            assert pm.dynamic_owner() == "Pro-BW"

    def test_test_promote_owner(self):
        meta = {"mode": "test_promote", "model_name": "m-260222", "endpoint_name": "e-1-test", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("MB")
            assert pm.dynamic_owner() == "Pro-Test-MB"


class TestPipelineMetaRepr:
    """Tests for PipelineMeta string representation."""

    def test_repr_with_env_var(self):
        meta = {
            "mode": "dt",
            "model_name": "my-model-dt",
            "endpoint_name": "my-endpoint-dt",
            "serverless": True,
        }
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            r = repr(pm)
            assert "PipelineMeta" in r
            assert "mode=dt" in r
            assert "my-model-dt" in r
