"""Tests for sizing a real-time endpoint instance from a model's measured serving memory"""

import pytest

from workbench.core.transforms.model_to_endpoint.model_to_endpoint import realtime_instance_for_memory


@pytest.mark.parametrize(
    "memory_gb, instance",
    [
        (1.5, "ml.c7i.large"),
        (3.0, "ml.c7i.large"),
        (3.13, "ml.m7i.large"),
        (9.17, "ml.r7i.large"),
        (12.0, "ml.r7i.large"),
        (12.1, "ml.r7i.xlarge"),
        (40.0, "ml.r7i.2xlarge"),
    ],
)
def test_smallest_instance_that_fits(memory_gb, instance):
    assert realtime_instance_for_memory(memory_gb) == instance


def test_past_the_ladder_raises():
    with pytest.raises(ValueError, match="pass instance="):
        realtime_instance_for_memory(100.0)
