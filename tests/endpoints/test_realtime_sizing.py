"""Tests for sizing an endpoint from a model's measured serving memory"""

import logging
from types import SimpleNamespace

import pytest

from workbench.api import ModelFramework
from workbench.core.artifacts.endpoint_core import EndpointCapacityError
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import (
    CAPACITY_ATTEMPTS,
    GPU_ASYNC_INSTANCE_POOLS,
    GPU_ASYNC_MAX_INSTANCES,
    GPU_INSTANCE_POOLS,
    SERVERLESS_MODEL_MEMORY_LIMIT_GB,
    ModelToEndpoint,
    realtime_instance,
)


def _model(memory_gb):
    """A stand-in for a ModelCore carrying (or not) a measured serving memory."""
    meta = {} if memory_gb is None else {"workbench_inference_memory_gb": memory_gb}
    return SimpleNamespace(name="test-model", workbench_meta=lambda: meta)


@pytest.mark.parametrize(
    "memory_gb, instance",
    [
        (None, "ml.c7i.large"),
        (1.5, "ml.c7i.large"),
        (3.0, "ml.c7i.large"),
        (3.13, "ml.m7i.large"),
        (9.17, "ml.r7i.large"),
        (12.0, "ml.r7i.large"),
        (12.1, "ml.r7i.xlarge"),
        (40.0, "ml.r7i.2xlarge"),
    ],
)
def test_realtime_default_until_the_model_outgrows_it(memory_gb, instance):
    assert realtime_instance(memory_gb) == instance


@pytest.mark.parametrize(
    "memory_gb, instance",
    [(None, "ml.c7i.xlarge"), (5.0, "ml.c7i.xlarge"), (6.1, "ml.r7i.large"), (20.0, "ml.r7i.xlarge")],
)
def test_async_keeps_its_larger_default(memory_gb, instance):
    assert realtime_instance(memory_gb, async_endpoint=True) == instance


def test_past_the_ladder_raises():
    with pytest.raises(ValueError, match="pass instance="):
        realtime_instance(100.0)


def test_serverless_refused_over_the_limit():
    transform = SimpleNamespace(serverless=True)
    with pytest.raises(ValueError, match="serverless=False"):
        ModelToEndpoint._check_serverless_memory(transform, _model(SERVERLESS_MODEL_MEMORY_LIMIT_GB + 0.01))


@pytest.mark.parametrize("memory_gb", [None, 2.0, SERVERLESS_MODEL_MEMORY_LIMIT_GB])
def test_serverless_allowed_at_or_under_the_limit(memory_gb):
    ModelToEndpoint._check_serverless_memory(SimpleNamespace(serverless=True), _model(memory_gb))


def test_realtime_is_never_refused():
    ModelToEndpoint._check_serverless_memory(SimpleNamespace(serverless=False), _model(50.0))


@pytest.mark.parametrize("framework", [ModelFramework.ATOMJEPA])
def test_gpu_frameworks_refuse_serverless(framework):
    model = SimpleNamespace(name="test-model", model_framework=framework)
    with pytest.raises(ValueError, match="async_endpoint=True"):
        ModelToEndpoint._check_serverless_framework(SimpleNamespace(serverless=True), model)
    ModelToEndpoint._check_serverless_framework(SimpleNamespace(serverless=False), model)


@pytest.mark.parametrize("framework", [ModelFramework.CHEMPROP, ModelFramework.TABICL, ModelFramework.XGBOOST])
def test_other_frameworks_may_serve_serverless(framework):
    model = SimpleNamespace(name="test-model", model_framework=framework)
    ModelToEndpoint._check_serverless_framework(SimpleNamespace(serverless=True), model)


def _transform(async_endpoint, max_instances, max_instances_given):
    return SimpleNamespace(
        async_endpoint=async_endpoint, max_instances=max_instances, max_instances_given=max_instances_given
    )


@pytest.mark.parametrize("pools", [GPU_INSTANCE_POOLS, GPU_ASYNC_INSTANCE_POOLS])
def test_every_instance_pool_is_a_gpu(pools):
    """A CPU pool here would be taken rather than waited out.

    SageMaker serves from the highest-priority pool that has capacity, and a CPU instance
    always has capacity, so one in this list resolves a GPU shortage in seconds at roughly
    50x the per-row cost instead of spending GPU_PROVISION_TIMEOUT_SECONDS on a GPU.
    """
    assert 1 <= len(pools) <= 5
    families = [pool.split(".")[1] for pool in pools]
    assert all(family.startswith(("g", "p")) for family in families), families


def test_only_async_pools_hold_a_t4():
    """A T4 serves at ~2 s/molecule: fine for an async invocation, past a real-time one's 60 s."""
    assert not any(".g4dn." in pool for pool in GPU_INSTANCE_POOLS)
    assert GPU_ASYNC_INSTANCE_POOLS[-1] == "ml.g4dn.xlarge"


def test_gpu_async_defaults_to_one_instance():
    transform = _transform(async_endpoint=True, max_instances=8, max_instances_given=False)
    ModelToEndpoint._apply_gpu_defaults(transform, SimpleNamespace(model_framework=ModelFramework.ATOMJEPA))
    assert transform.max_instances == GPU_ASYNC_MAX_INSTANCES


def test_gpu_async_keeps_a_caller_bound():
    transform = _transform(async_endpoint=True, max_instances=4, max_instances_given=True)
    ModelToEndpoint._apply_gpu_defaults(transform, SimpleNamespace(model_framework=ModelFramework.ATOMJEPA))
    assert transform.max_instances == 4


def test_cpu_async_keeps_its_default():
    transform = _transform(async_endpoint=True, max_instances=8, max_instances_given=False)
    ModelToEndpoint._apply_gpu_defaults(transform, SimpleNamespace(model_framework=ModelFramework.CHEMPROP))
    assert transform.max_instances == 8


def _deployer(failures: int) -> SimpleNamespace:
    """A transform whose deploy finds no capacity for the first `failures` attempts."""
    calls = []

    def deploy(model, **kwargs):
        calls.append(model)
        if len(calls) <= failures:
            raise EndpointCapacityError("no capacity")

    return SimpleNamespace(_deploy_model=deploy, output_name="test-endpoint", log=logging.getLogger(), calls=calls)


def test_capacity_failure_is_retried_until_it_deploys():
    transform = _deployer(failures=CAPACITY_ATTEMPTS - 1)
    ModelToEndpoint._deploy_with_capacity_retries(transform, "model")
    assert len(transform.calls) == CAPACITY_ATTEMPTS


def test_capacity_failure_raises_after_the_last_attempt():
    transform = _deployer(failures=CAPACITY_ATTEMPTS)
    with pytest.raises(EndpointCapacityError):
        ModelToEndpoint._deploy_with_capacity_retries(transform, "model")
    assert len(transform.calls) == CAPACITY_ATTEMPTS
