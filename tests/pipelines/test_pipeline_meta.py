"""Tests for the PipelineMeta class"""

import json
import os
from unittest.mock import patch

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

    def test_missing_keys_get_defaults(self):
        """Partial PIPELINE_META should fill in defaults for missing keys."""
        meta = {"mode": "dt"}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "dt"
            assert pm.serverless is True
            assert pm.model_name.startswith("test-")
            assert pm.endpoint_name.startswith("test-")

    def test_invalid_json_falls_back_to_defaults(self):
        with patch.dict(os.environ, {"PIPELINE_META": "not-valid-json"}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "dev"
            assert pm.model_name.startswith("test-")
            assert pm.serverless is True


class TestPipelineMetaDefaults:
    """Tests for PipelineMeta when no PIPELINE_META env var is set."""

    def test_default_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            assert pm.mode == "dev"

    def test_default_names_have_timestamp(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            assert pm.model_name.startswith("test-")
            assert pm.endpoint_name.startswith("test-")
            # Should have format test-YYYYMMDD-HHMM
            assert len(pm.model_name) == len("test-20260222-0947")

    def test_default_serverless(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            assert pm.serverless is True

    def test_get_with_default(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            assert pm.get("nonexistent") is None
            assert pm.get("nonexistent", "fallback") == "fallback"


class TestPipelineMetaOwner:
    """Tests for set_owner() and dynamic_owner()."""

    def test_dt_mode_owner(self):
        meta = {"mode": "dt", "model_name": "m-dt", "endpoint_name": "e-dt", "serverless": True}
        with patch.dict(os.environ, {"PIPELINE_META": json.dumps(meta)}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("BW")
            assert pm.dynamic_owner() == "DT"

    def test_temporal_split_owner(self):
        meta = {"mode": "temporal_split", "model_name": "m-ts", "endpoint_name": "e-ts", "serverless": True}
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

    def test_dev_mode_owner(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            pm.set_owner("ER")
            assert pm.dynamic_owner() == "ER"

    def test_dynamic_owner_without_set_owner_defaults_to_test(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            assert pm.dynamic_owner() == "test"


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

    def test_repr_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            pm = PipelineMeta()
            r = repr(pm)
            assert "PipelineMeta" in r
            assert "mode=dev" in r
