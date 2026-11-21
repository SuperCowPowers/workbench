"""Unit tests for how an endpoint sizes the calls made to it.

Local + fast (no AWS). Each endpoint owns its rows per invocation and its invocations at
once, and every caller reads them: a direct client, or a MetaEndpoint calling sync and async
children. The queue has to match what the fleet drains: too deep and chunks die on a poll
deadline they spent waiting rather than working.
"""

import logging
import threading
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from workbench.core.artifacts import async_endpoint_core, endpoint_core
from workbench.core.artifacts.async_endpoint_core import AsyncEndpointCore
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import (
    GPU_BATCH_SIZE,
    GPU_REALTIME_BATCH_SIZE,
    GPU_REALTIME_CONCURRENCY,
    default_batch_size,
    default_concurrency,
)
from workbench.endpoints.fast_inference import SYNC_BATCH_SIZE, SYNC_CONCURRENCY, fast_inference, runtime_client
from workbench.utils import async_endpoint_utils as au
from workbench.utils.meta_endpoint_dag import SYNC_META_BATCH_SIZE, MetaEndpointDAG


@pytest.fixture
def workbench_logs(caplog, monkeypatch):
    """Capture the `workbench` logger, which sets propagate False so caplog misses it."""
    monkeypatch.setattr(logging.getLogger("workbench"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="workbench")
    return caplog


@pytest.mark.parametrize(
    "instance_type, vcpus",
    [
        ("ml.c7i.xlarge", 4),
        ("ml.c7i.4xlarge", 16),
        ("ml.g6.xlarge", 4),
        ("ml.g6.2xlarge", 8),
        ("ml.m5.large", 2),
        ("ml.t2.medium", 1),
        ("ml.c7i.metal-24xl", 0),  # outside the pattern: caller keeps its default
    ],
)
def test_instance_vcpus_reads_the_size_suffix(instance_type, vcpus):
    assert au.instance_vcpus(instance_type) == vcpus


@pytest.mark.parametrize(
    "instance_type, pools, async_endpoint, expected",
    [
        (None, ["ml.g6.xlarge", "ml.g5.xlarge"], True, GPU_BATCH_SIZE),
        (None, ["ml.g6.xlarge", "ml.g5.xlarge"], False, GPU_REALTIME_BATCH_SIZE),  # 60 s to answer
        ("ml.c7i.4xlarge", None, True, 32),  # an explicit CPU instance fills its vCPUs
        ("ml.c7i.metal-24xl", None, True, au.ASYNC_BATCH_SIZE),
        ("ml.c7i.large", None, False, SYNC_BATCH_SIZE),
        (None, None, False, SYNC_BATCH_SIZE),  # serverless
    ],
)
def test_default_batch_size(instance_type, pools, async_endpoint, expected):
    assert default_batch_size(instance_type, pools, async_endpoint) == expected


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({"inference_batch_size": 4}, 4),
        ({"async_endpoint": True, "inference_batch_size": 64}, 64),
        ({"async_endpoint": True}, au.ASYNC_BATCH_SIZE),  # deployed before stamping
        ({}, SYNC_BATCH_SIZE),
    ],
)
def test_endpoint_reads_its_own_batch_size(meta, expected):
    endpoint = SimpleNamespace(workbench_meta=lambda: meta)
    assert EndpointCore.inference_batch_size(endpoint) == expected


def test_fast_inference_runs_at_the_batch_size():
    """A slow sync endpoint gets the small chunks it stamped."""
    runtime = MagicMock()
    runtime.invoke_endpoint.side_effect = lambda **kwargs: {"Body": BytesIO(kwargs["Body"].encode())}
    session = MagicMock()
    session.boto_session.client.return_value = runtime

    result = fast_inference("ep", pd.DataFrame({"x": range(11)}), session, batch_size=5)

    assert runtime.invoke_endpoint.call_count == 3
    assert len(result) == 11


def test_realtime_gpu_takes_one_invocation_at_a_time():
    assert default_concurrency(["ml.g6.xlarge", "ml.g5.xlarge"]) == GPU_REALTIME_CONCURRENCY == 1
    assert default_concurrency(None) == SYNC_CONCURRENCY


SERVERLESS_VARIANT = {"ProductionVariants": [{"CurrentServerlessConfig": {"MaxConcurrency": 2}}]}


@pytest.mark.parametrize(
    "meta, endpoint_meta, expected",
    [
        ({"inference_concurrency": 1}, {}, 1),  # a real-time GPU endpoint's stamp
        ({}, {}, SYNC_CONCURRENCY),  # a real-time endpoint without one
        ({"serverless": True}, SERVERLESS_VARIANT, 2),
        ({"serverless": True, "inference_concurrency": 1}, SERVERLESS_VARIANT, 1),  # a stamp lowers it
        ({"serverless": True, "inference_concurrency": 8}, SERVERLESS_VARIANT, 2),  # never past MaxConcurrency
        ({"async_endpoint": True, "max_instances": 8}, {}, 8 * au.IN_FLIGHT_PER_SLOT),
        ({"async_endpoint": True, "max_instances": 1}, {}, au.IN_FLIGHT_PER_SLOT),
    ],
)
def test_endpoint_reads_its_own_concurrency(meta, endpoint_meta, expected):
    endpoint = SimpleNamespace(workbench_meta=lambda: meta, endpoint_meta=endpoint_meta)
    assert EndpointCore.inference_concurrency(endpoint) == expected


def test_endpoint_fast_inference_runs_at_its_own_sizing(monkeypatch):
    """The endpoint, not the caller, decides rows per invocation and invocations at once."""
    sent = {}
    monkeypatch.setattr(endpoint_core, "fast_inference", lambda *args, **kwargs: sent.update(kwargs))
    endpoint = MagicMock()
    endpoint.inference_batch_size.return_value = 16
    endpoint.inference_concurrency.return_value = 1

    EndpointCore.fast_inference(endpoint, pd.DataFrame({"x": [1]}))

    assert sent == {"threads": 1, "batch_size": 16}


def test_fast_inference_keeps_no_more_than_threads_outstanding():
    """One thread means the next invocation leaves only after the last one answered."""
    outstanding, peak = 0, 0
    lock = threading.Lock()

    def invoke_endpoint(**kwargs):
        nonlocal outstanding, peak
        with lock:
            outstanding += 1
            peak = max(peak, outstanding)
        time.sleep(0.01)
        with lock:
            outstanding -= 1
        return {"Body": BytesIO(kwargs["Body"].encode())}

    runtime = MagicMock()
    runtime.invoke_endpoint.side_effect = invoke_endpoint
    session = MagicMock()
    session.boto_session.client.return_value = runtime

    fast_inference("ep", pd.DataFrame({"x": range(8)}), session, threads=1, batch_size=2)

    assert runtime.invoke_endpoint.call_count == 4
    assert peak == 1


def test_runtime_client_outlasts_the_server_invocation_limit():
    """A read timeout at the server's 60 s would re-send a slow chunk before its error arrives."""
    made = {}
    session = SimpleNamespace(client=lambda service, config: made.update(service=service, config=config))

    runtime_client(session)

    assert made["service"] == "sagemaker-runtime"
    assert made["config"].read_timeout > 60


def test_inference_chunks_through_the_runtime_client(monkeypatch):
    """inference() and fast_inference() invoke the same way."""
    runtime = MagicMock()
    runtime.invoke_endpoint.side_effect = lambda **kwargs: {"Body": BytesIO(kwargs["Body"].encode())}
    monkeypatch.setattr(endpoint_core, "runtime_client", lambda session: runtime)
    core = object.__new__(EndpointCore)  # skip __init__ (no AWS)
    core.name = "ep"
    core.boto3_session = None  # shadows the lazy class attribute, which would build an AWS session
    core.log = logging.getLogger("test")
    core.endpoint_return_columns = None
    core.inference_batch_size = lambda: 2

    out = core._predict(pd.DataFrame({"x": range(5)}), ["x"])

    assert runtime.invoke_endpoint.call_count == 3
    assert out["x"].tolist() == list(range(5))


def test_in_flight_follows_the_fleet_ceiling():
    """A cold endpoint sizes on what it can scale to, so the queue reads as demand."""
    assert au.async_max_in_flight({"max_instances": 8}) == 8 * au.IN_FLIGHT_PER_SLOT


def test_in_flight_multiplies_fleet_by_per_instance():
    meta = {"max_instances": 3, "async_max_concurrent": 4}
    assert au.async_max_in_flight(meta) == 12 * au.IN_FLIGHT_PER_SLOT


def test_in_flight_without_stamps_is_one_slot():
    """An endpoint with no stamps costs throughput, never correctness."""
    assert au.async_max_in_flight({}) == au.IN_FLIGHT_PER_SLOT


def test_async_client_sends_what_the_endpoint_stamped(monkeypatch):
    """The stamped batch and the fleet's queue depth both reach async_inference."""
    sent = {}
    monkeypatch.setattr(async_endpoint_core, "async_inference", lambda **kwargs: sent.update(kwargs))
    meta = {"async_endpoint": True, "inference_batch_size": 64, "max_instances": 1}
    endpoint = MagicMock(endpoint_meta={"EndpointConfigName": "cfg", "ProductionVariants": [{}]})
    endpoint.boto3_session.client.return_value.describe_endpoint_config.return_value = {"ProductionVariants": [{}]}
    endpoint.workbench_meta.return_value = meta
    endpoint.inference_batch_size.return_value = 64
    endpoint.inference_concurrency = lambda: EndpointCore.inference_concurrency(endpoint)

    AsyncEndpointCore._async_batch_invoke(endpoint, pd.DataFrame({"x": [1]}))

    assert sent["batch_size"] == 64
    assert sent["max_in_flight"] == au.IN_FLIGHT_PER_SLOT


def child_meta(monkeypatch, children: dict) -> MetaEndpointDAG:
    """A DAG over children given as {endpoint_name: workbench_meta}, with their metadata read."""

    class FakeEndpoint:
        def __init__(self, name):
            self.endpoint_meta = SERVERLESS_VARIANT if children[name].get("serverless") else {}
            self.workbench_meta = lambda: children[name]
            self.inference_batch_size = lambda: EndpointCore.inference_batch_size(self)
            self.inference_concurrency = lambda: EndpointCore.inference_concurrency(self)

    monkeypatch.setattr("workbench.api.Endpoint", FakeEndpoint)
    dag = MetaEndpointDAG()
    for name in children:
        dag.add_endpoint(name, ["target"])
    dag.populate_child_metadata()
    return dag


def test_mixed_meta_calls_each_child_at_its_own_size(monkeypatch):
    """An async meta over a sync and an async child reads each child's stamp."""
    dag = child_meta(
        monkeypatch,
        {
            "fast-2d": {"inference_batch_size": 200},
            "slow-3d": {"async_endpoint": True, "inference_batch_size": 4, "max_instances": 8},
        },
    )

    assert dag.endpoint_batch_sizes == {"fast-2d": 200, "slow-3d": 4}
    assert dag.endpoint_concurrency == {"fast-2d": SYNC_CONCURRENCY, "slow-3d": 16}


def test_meta_calls_every_kind_of_child_at_its_own_concurrency(monkeypatch):
    """Serverless, real-time GPU and async children each answer through the same accessor."""
    dag = child_meta(
        monkeypatch,
        {
            "serverless": {"serverless": True},
            "gpu-realtime": {"inference_batch_size": 16, "inference_concurrency": 1},
            "gpu-async": {"async_endpoint": True, "inference_batch_size": 64, "max_instances": 1},
        },
    )

    assert dag.endpoint_concurrency == {"serverless": 2, "gpu-realtime": 1, "gpu-async": au.IN_FLIGHT_PER_SLOT}


def test_meta_reads_unstamped_children_at_their_defaults(monkeypatch):
    """A child deployed before stamping can still sit under a meta."""
    dag = child_meta(monkeypatch, {"sync": {}, "async": {"async_endpoint": True}})

    assert dag.endpoint_batch_sizes == {"sync": SYNC_BATCH_SIZE, "async": au.ASYNC_BATCH_SIZE}


def test_meta_sizing_survives_serialization(monkeypatch):
    """The deployed container rebuilds the DAG from its dict."""
    dag = child_meta(monkeypatch, {"child": {"async_endpoint": True, "inference_batch_size": 4, "max_instances": 8}})

    rebuilt = MetaEndpointDAG.from_dict(dag.to_dict())

    assert rebuilt.endpoint_batch_sizes == {"child": 4}
    assert rebuilt.endpoint_concurrency == {"child": 16}


def test_meta_batch_sizes_to_the_slowest_child(monkeypatch):
    """Four rounds through the smallest-batch child's own slots, not another child's fleet."""
    dag = child_meta(
        monkeypatch,
        {
            "slow": {"async_endpoint": True, "inference_batch_size": 4, "max_instances": 2},
            "fast": {"async_endpoint": True, "inference_batch_size": 64, "max_instances": 8},
            "sync": {"inference_batch_size": 100},
        },
    )

    assert dag.meta_batch_size() == 4 * 2 * 4


GPU_REALTIME = {"inference_batch_size": 16, "inference_concurrency": 1}


@pytest.mark.parametrize(
    "children, batch, concurrency",
    [
        ({"a": {}, "b": {}}, 400, SYNC_CONCURRENCY),  # real-time children: 100 rows x 4 at once
        ({"a": {"serverless": True}, "b": {"serverless": True}}, 200, 2),
        ({"fast": {"serverless": True}, "gpu": GPU_REALTIME}, 16, 1),  # one 16-row call, inside 60 s
        ({"a": {"inference_batch_size": 400}, "b": {"inference_batch_size": 200}}, SYNC_META_BATCH_SIZE, 4),
    ],
)
def test_sync_meta_downshifts_to_its_tightest_child(monkeypatch, children, batch, concurrency):
    """A sync meta has 60 s, so it takes one round through the child that takes the fewest rows at once."""
    dag = child_meta(monkeypatch, children)

    assert dag.meta_batch_size() == batch
    assert dag.meta_concurrency() == concurrency


def test_async_meta_takes_its_concurrency_from_its_own_queue(monkeypatch):
    dag = child_meta(monkeypatch, {"slow": {"async_endpoint": True, "max_instances": 2}, "gpu": GPU_REALTIME})

    assert dag.meta_concurrency() is None


def pool_pair(preferred: str, fallback: str, serving: str) -> tuple:
    """An endpoint + config describing a two-pool variant, with one pool up."""
    endpoint = {
        "ProductionVariants": [
            {
                "CurrentInstanceCount": 1,
                "InstancePools": [
                    {"InstanceType": preferred, "CurrentInstanceCount": 1 if serving == preferred else 0},
                    {"InstanceType": fallback, "CurrentInstanceCount": 1 if serving == fallback else 0},
                ],
            }
        ]
    }
    config = {
        "ProductionVariants": [
            {"InstancePools": [{"InstanceType": fallback, "Priority": 2}, {"InstanceType": preferred, "Priority": 1}]}
        ]
    }
    return endpoint, config


def test_fallback_pool_is_warned_about(workbench_logs):
    """Serving off the fallback is a performance cliff the endpoint status hides."""
    endpoint, config = pool_pair("ml.g6.xlarge", "ml.c7i.4xlarge", serving="ml.c7i.4xlarge")

    au.warn_on_fallback_pool("ep", endpoint, config)

    assert "ml.c7i.4xlarge" in workbench_logs.text and "ml.g6.xlarge" in workbench_logs.text


def test_preferred_pool_is_quiet(workbench_logs):
    """Serving the first choice says nothing."""
    endpoint, config = pool_pair("ml.g6.xlarge", "ml.c7i.4xlarge", serving="ml.g6.xlarge")

    au.warn_on_fallback_pool("ep", endpoint, config)

    assert workbench_logs.text == ""


def test_single_pool_variant_is_quiet(workbench_logs):
    """With nothing to fall back to there is no fallback to report."""
    endpoint = {
        "ProductionVariants": [{"InstancePools": [{"InstanceType": "ml.c7i.xlarge", "CurrentInstanceCount": 1}]}]
    }
    config = {"ProductionVariants": [{"InstancePools": [{"InstanceType": "ml.c7i.xlarge", "Priority": 1}]}]}

    au.warn_on_fallback_pool("ep", endpoint, config)

    assert workbench_logs.text == ""
