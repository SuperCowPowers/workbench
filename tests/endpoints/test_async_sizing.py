"""Unit tests for sizing an async batch against the endpoint it is going to.

Local + fast (no AWS). Two numbers come off the endpoint rather than a constant: the
batch has to fill the serving box's thread pool, and the queue has to match what the
fleet drains. Queue too deep and chunks die on a poll deadline they spent waiting
rather than working; batch too narrow and the container idles on vCPUs it owns.
"""

import logging
from unittest.mock import MagicMock

import pytest

from workbench.utils import async_endpoint_utils as au

CAP = au.AsyncCapacity


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


def test_batch_fills_the_serving_box():
    """A 16-vCPU box takes two rows per vCPU."""
    batch_size, _ = au.resolve_batch_sizing({}, n_rows=750, capacity=CAP(concurrency=1, vcpus=16))
    assert batch_size == 32


def test_batch_falls_back_when_the_box_is_unknown():
    """An instance type with no vCPU count leaves the default in place."""
    batch_size, _ = au.resolve_batch_sizing({}, n_rows=750, capacity=CAP(concurrency=1, vcpus=0))
    assert batch_size == au.DEFAULT_BATCH_SIZE


def test_batch_override_wins():
    """A measured value beats the derived one, for an endpoint whose rows are too slow."""
    batch_size, _ = au.resolve_batch_sizing(
        {"inference_batch_size": 5}, n_rows=750, capacity=CAP(concurrency=1, vcpus=16)
    )
    assert batch_size == 5


def test_in_flight_follows_endpoint_concurrency():
    """A serial endpoint gets a shallow queue, not one worker per chunk."""
    _, in_flight = au.resolve_batch_sizing({}, n_rows=750, capacity=CAP(concurrency=1, vcpus=16))
    assert in_flight == au.IN_FLIGHT_PER_SLOT  # 24 chunks are available; we send 2


def test_in_flight_scales_with_concurrency():
    """Four slots earn four times the queue depth."""
    _, serial = au.resolve_batch_sizing({}, n_rows=750, capacity=CAP(concurrency=1, vcpus=16))
    _, parallel = au.resolve_batch_sizing({}, n_rows=750, capacity=CAP(concurrency=4, vcpus=16))
    assert parallel == serial * 4


def test_in_flight_never_exceeds_chunk_count():
    """Plenty of slots and only two chunks still submits two."""
    _, in_flight = au.resolve_batch_sizing({}, n_rows=64, capacity=CAP(concurrency=16, vcpus=16))
    assert in_flight == 2


def test_in_flight_override_wins():
    _, in_flight = au.resolve_batch_sizing(
        {"inference_max_in_flight": 7}, n_rows=750, capacity=CAP(concurrency=1, vcpus=16)
    )
    assert in_flight == 7


def fake_session(current: int, per_instance: int, pools: list = None, instance_type: str = None) -> MagicMock:
    """A boto session whose SageMaker client describes one async endpoint."""
    variant = {"InstanceType": instance_type}
    if pools is not None:
        variant["InstancePools"] = [{"InstanceType": t, "Priority": i} for i, t in enumerate(pools, 1)]
    sm = MagicMock()
    sm.describe_endpoint.return_value = {
        "ProductionVariants": [{"CurrentInstanceCount": current}],
        "EndpointConfigName": "cfg",
    }
    sm.describe_endpoint_config.return_value = {
        "ProductionVariants": [variant],
        "AsyncInferenceConfig": {"ClientConfig": {"MaxConcurrentInvocationsPerInstance": per_instance}},
    }
    session = MagicMock()
    session.client.return_value = sm
    return session


def test_capacity_uses_the_fleet_ceiling(monkeypatch):
    """A cold endpoint sizes on what it can scale to, not on the one instance up.

    Sizing on the current count would leave the queue too shallow to read as demand,
    so the endpoint would never be scaled out and would stay at that one instance.
    """
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: fake_session(1, 1, instance_type="ml.c7i.xlarge"))

    assert au.async_capacity("ep", {"max_instances": 8}).concurrency == 8


def test_capacity_multiplies_fleet_by_per_instance(monkeypatch):
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: fake_session(1, 4, instance_type="ml.c7i.xlarge"))

    assert au.async_capacity("ep", {"max_instances": 3}).concurrency == 12


def test_capacity_without_a_ceiling_uses_current_count(monkeypatch):
    """An endpoint with no stamped ceiling falls back to what is actually running."""
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: fake_session(2, 1, instance_type="ml.c7i.xlarge"))

    assert au.async_capacity("ep", {}).concurrency == 2


def test_capacity_takes_the_largest_pool(monkeypatch):
    """Instance pools mean the serving type is not known until one comes up.

    The widest is the safe choice: a batch wider than the box only lengthens an
    invocation, while a narrower one leaves the container's pool starved.
    """
    pools = ["ml.g6.xlarge", "ml.g5.xlarge", "ml.c7i.4xlarge"]
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: fake_session(1, 1, pools=pools))

    assert au.async_capacity("ep", {}).vcpus == 16


def test_capacity_treats_an_unset_container_cap_as_one(monkeypatch):
    """SageMaker picks its own value when the cap is unset and never reports it back."""
    session = fake_session(1, 1, instance_type="ml.c7i.xlarge")
    session.client.return_value.describe_endpoint_config.return_value = {
        "ProductionVariants": [{"InstanceType": "ml.c7i.xlarge"}],
        "AsyncInferenceConfig": {"ClientConfig": {}},
    }
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: session)

    assert au.async_capacity("ep", {"max_instances": 4}).concurrency == 4


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


def test_capacity_falls_back_to_one_slot(monkeypatch):
    """An unreadable endpoint costs throughput, never correctness."""
    session = MagicMock()
    session.client.return_value.describe_endpoint.side_effect = RuntimeError("no such endpoint")
    monkeypatch.setattr(au, "resolve_boto_session", lambda s: session)

    assert au.async_capacity("ep", {}) == CAP(1, 0)
