"""Auto-scaling utilities for SageMaker endpoints (async and realtime)."""

import logging

log = logging.getLogger("workbench")

# Standard scaling metrics for SageMaker endpoints
ASYNC_METRIC = {
    "MetricName": "HasBacklogPerInstance",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Average",
}
REALTIME_METRIC = {
    "MetricName": "InvocationsPerInstance",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Sum",
}


def register_autoscaling(
    boto3_session,
    endpoint_name: str,
    min_capacity: int = 0,
    max_capacity: int = 4,
    scale_in_cooldown: int = 300,
    scale_out_cooldown: int = 60,
) -> None:
    """Register an auto-scaling policy for a SageMaker endpoint.

    Automatically selects the appropriate scaling metric:
      - min_capacity=0 → async (HasBacklogPerInstance, target=1.0)
      - min_capacity>=1 → realtime (InvocationsPerInstance, target=750.0)

    Args:
        boto3_session: Boto3 session for API calls
        endpoint_name: Name of the SageMaker endpoint
        min_capacity: Minimum instance count (0 = scale to zero for async)
        max_capacity: Maximum instance count under load
        scale_in_cooldown: Seconds to wait before scaling in (default 300)
        scale_out_cooldown: Seconds to wait before scaling out (default 60)
    """
    aas_client = boto3_session.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Pick metric and target based on whether this is async (min=0) or realtime
    if min_capacity == 0:
        metric = ASYNC_METRIC
        target_value = 1.0
    else:
        metric = REALTIME_METRIC
        target_value = 750.0

    try:
        aas_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

        aas_client.put_scaling_policy(
            PolicyName=f"{endpoint_name}-autoscaling",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": target_value,
                "CustomizedMetricSpecification": {
                    **metric,
                    "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                },
                "ScaleInCooldown": scale_in_cooldown,
                "ScaleOutCooldown": scale_out_cooldown,
            },
        )
        log.important(
            f"Auto-scaling registered for {endpoint_name}: "
            f"min={min_capacity}, max={max_capacity}, metric={metric['MetricName']}"
        )
    except Exception as e:
        log.warning(f"Failed to register auto-scaling for {endpoint_name}: {e}")
