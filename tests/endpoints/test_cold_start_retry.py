"""Unit tests for cold-start handling on endpoints with serverless children.

Local + fast (no AWS). A serverless child that is cold can take longer than
SageMaker's 60s invocation limit to come up, which surfaces as a ModelError with
"could not get a response". These tests pin the chain that turns that into a
batch retry: ``fast_inference`` propagates the child's error text, and the client
recognises it through a MetaEndpoint's 500 instead of bisecting down to NaN rows.
"""

import json
import logging
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from workbench.core.artifacts import endpoint_core as mod
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.endpoints.fast_inference import fast_inference

_COLD_CHILD = (
    'Received server error (0) from model with message "Amazon SageMaker could not get a '
    'response from the cl-mouse-reg-1 endpoint.".'
)


def _model_error(message: str) -> ClientError:
    return ClientError({"Error": {"Code": "ModelError", "Message": message}}, "InvokeEndpoint")


class _FailingRuntime:
    def invoke_endpoint(self, **kwargs):
        raise _model_error(_COLD_CHILD)


def test_fast_inference_propagates_the_child_error():
    session = SimpleNamespace(client=lambda *args, **kwargs: _FailingRuntime())
    with pytest.raises(ClientError, match="could not get a response"):
        fast_inference("cl-mouse-reg-1", pd.DataFrame({"smiles": ["CCO"]}), sm_session=session)


class _ColdThenWarm:
    """Stands in for a MetaEndpoint: the first call wraps a cold child's timeout in
    the inference server's 500 body, later calls succeed."""

    def __init__(self):
        self.calls = []

    def invoke(self, body, content_type, accept):
        rows = pd.read_csv(StringIO(body))
        self.calls.append(len(rows))
        if len(self.calls) == 1:
            meta_body = json.dumps({"error": f"An error occurred (ModelError): {_COLD_CHILD}"})
            raise _model_error(f'Received server error (500) from model with message "{meta_body}".')
        return SimpleNamespace(body=rows.assign(prediction=1.0))


def test_meta_wrapped_cold_start_retries_whole_batch(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    core = object.__new__(EndpointCore)  # skip __init__ (no AWS)
    core.log = logging.getLogger("test")
    core.endpoint_return_columns = None

    endpoint = _ColdThenWarm()
    out = core._endpoint_error_handling(endpoint, pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"]}))

    assert endpoint.calls == [3, 3]  # retried the batch; never bisected
    assert out["prediction"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
