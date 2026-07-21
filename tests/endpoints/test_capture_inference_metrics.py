"""Unit tests for the metrics-write guard in EndpointCore._capture_inference_results.

Local + fast (no AWS) — awswrangler is mocked. An empty metrics frame (a target
with no ground truth in the eval set) must NOT be written: pandas serializes an
empty DataFrame to a 1-byte CSV that read_csv later rejects with EmptyDataError.
The guard skips the metrics file in that case while still writing meta + predictions.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from workbench.api import ModelType
from workbench.core.artifacts.endpoint_core import EndpointCore

PRED_DF = pd.DataFrame({"id": [1, 2], "smiles": ["C", "CC"], "logd_pred": [0.1, 0.2]})
COMMON = dict(target="logd", model_type=ModelType.REGRESSOR, description="d", features=["smiles"], id_column="id")


def _fake_endpoint():
    """An EndpointCore with __init__ bypassed and only the touched attrs stubbed."""
    ep = EndpointCore.__new__(EndpointCore)
    ep.log = MagicMock()
    ep.endpoint_inference_path = "s3://fake-bucket/endpoints/fake/inference"
    ep._hash_dataframe = lambda df: "deadbeef"
    ep._save_target_inference = MagicMock()
    return ep


def _capture(metrics):
    """Run one capture with the given metrics frame; return the patched `wr`."""
    ep = _fake_endpoint()
    with patch("workbench.core.artifacts.endpoint_core.wr") as wr_mock:
        ep._capture_inference_results("pxr_phase1_test_logd", PRED_DF, metrics=metrics, **COMMON)
    return wr_mock


def test_empty_metrics_frame_skips_metrics_file():
    wr_mock = _capture(pd.DataFrame())
    assert wr_mock.s3.to_csv.call_count == 0  # no 1-byte inference_metrics.csv
    assert wr_mock.s3.to_json.call_count == 1  # meta.json still written


def test_none_metrics_skips_metrics_file():
    wr_mock = _capture(None)
    assert wr_mock.s3.to_csv.call_count == 0


def test_real_metrics_frame_writes_metrics_file():
    wr_mock = _capture(pd.DataFrame([{"rmse": 0.2, "r2": 0.96}]))
    assert wr_mock.s3.to_csv.call_count == 1
    path = wr_mock.s3.to_csv.call_args.args[1]
    assert path.endswith("pxr_phase1_test_logd/inference_metrics.csv")
