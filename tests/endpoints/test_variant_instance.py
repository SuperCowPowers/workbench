"""Tests for reporting the instance a serverless, single-type, or pooled endpoint serves from"""

from workbench.core.cloud_platform.aws.aws_meta import variant_instance
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import GPU_INSTANCE_POOLS

POOLS = [{"InstanceType": pool_type, "Priority": priority} for priority, pool_type in enumerate(GPU_INSTANCE_POOLS, 1)]


def test_serverless():
    config = {"ServerlessConfig": {"MemorySizeInMB": 6144, "MaxConcurrency": 2}}
    assert variant_instance(config) == "Serverless (6GB/2)"


def test_single_type():
    assert variant_instance({"InstanceType": "ml.c7i.large"}) == "ml.c7i.large"


def test_pooled_while_scaled_to_zero_lists_pools_by_priority():
    config = {"InstancePools": list(reversed(POOLS))}
    idle = {
        "InstancePools": [{"InstanceType": pool_type, "CurrentInstanceCount": 0} for pool_type in GPU_INSTANCE_POOLS]
    }
    expected = " > ".join(GPU_INSTANCE_POOLS)
    assert variant_instance(config) == expected
    assert variant_instance(config, idle) == expected


def test_pooled_reports_the_pool_serving():
    serving = {
        "InstancePools": [
            {"InstanceType": "ml.g6.xlarge", "CurrentInstanceCount": 0},
            {"InstanceType": "ml.g5.xlarge", "CurrentInstanceCount": 1},
        ]
    }
    assert variant_instance({"InstancePools": POOLS}, serving) == "ml.g5.xlarge"
