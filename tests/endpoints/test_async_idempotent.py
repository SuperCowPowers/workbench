"""Unit tests for idempotent async invocation (the MetaEndpoint redelivery fix).

Local + fast (no AWS) — S3/SageMaker clients are mocked. Verifies that
concurrent identical requests share one child compute (leader/follower) instead
of staging duplicate, never-completing child jobs.
"""

from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd

from workbench.endpoints import async_inference as ai


class _NoSuchKey(Exception):
    pass


def _s3_with_objects(objects):
    """Mock S3 client backed by a dict {key: bytes}; supports the calls we use."""
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = _NoSuchKey

    def get_object(Bucket, Key):
        if Key not in objects:
            raise _NoSuchKey()
        return {"Body": BytesIO(objects[Key])}

    def put_object(Bucket, Key, Body, **kwargs):
        # Emulate If-None-Match: * (create-if-absent) for the lock.
        if kwargs.get("IfNoneMatch") == "*" and Key in objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    s3.get_object.side_effect = get_object
    s3.put_object.side_effect = put_object
    return s3


DF = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})


def test_request_hash_is_deterministic():
    h1 = ai._request_hash("ep", DF.to_csv(index=False))
    h2 = ai._request_hash("ep", DF.to_csv(index=False))
    h3 = ai._request_hash("ep", pd.DataFrame({"smiles": ["CCN"]}).to_csv(index=False))
    h4 = ai._request_hash("other-ep", DF.to_csv(index=False))
    assert h1 == h2  # same input → same id
    assert h1 != h3 and h1 != h4  # content and endpoint both matter


def test_leader_invokes_child_once_and_returns(monkeypatch):
    """The leader (wins the lock) stages input, invokes, and returns the result."""
    objects = {}
    s3 = _s3_with_objects(objects)
    runtime = MagicMock()
    runtime.invoke_endpoint_async.return_value = {"OutputLocation": "s3://b/out/r.csv"}
    monkeypatch.setattr(ai, "_poll_s3_output", lambda _c, _loc: DF.to_csv(index=False))

    out = ai._invoke_one_async_idempotent(runtime, s3, "child", DF, "b", "endpoints/child/async-input")

    assert out.equals(DF)
    runtime.invoke_endpoint_async.assert_called_once()  # exactly one child compute


def test_follower_shares_leaders_output_without_reinvoking(monkeypatch):
    """A follower (lock already held, OutputLocation recorded) reuses it — no new invoke."""
    req_hash = ai._request_hash("child", DF.to_csv(index=False))
    lock_key = f"endpoints/child/async-idem/{req_hash}.lock"
    # Lock already exists with the leader's recorded OutputLocation.
    objects = {lock_key: b"s3://b/out/shared.csv"}
    s3 = _s3_with_objects(objects)
    runtime = MagicMock()
    monkeypatch.setattr(ai, "_poll_s3_output", lambda _c, loc: DF.to_csv(index=False))

    out = ai._invoke_one_async_idempotent(runtime, s3, "child", DF, "b", "endpoints/child/async-input")

    assert out.equals(DF)
    runtime.invoke_endpoint_async.assert_not_called()  # shared, no duplicate compute
