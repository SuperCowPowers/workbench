"""Auto-scaling utilities for SageMaker endpoints (async and realtime).

Async scale-to-zero requires TWO scaling policies because
``ApproximateBacklogSizePerInstance`` is undefined when the instance
count is 0 (divide-by-zero), so target-tracking alone cannot make the
0 → 1 transition:

    StepScaling    on HasBacklogWithoutCapacity            → 0 → 1
    TargetTracking on ApproximateBacklogSizePerInstance    → 1 → N
    (target-tracking also handles N → 0 scale-in)

Realtime endpoints use a single target-tracking policy on
``InvocationsPerInstance``.
"""

import logging
from typing import Optional

log = logging.getLogger("workbench")

# Metric definitions
_BACKLOG_PER_INSTANCE = {
    "MetricName": "ApproximateBacklogSizePerInstance",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Average",
}
_HAS_BACKLOG_WITHOUT_CAPACITY = {
    "MetricName": "HasBacklogWithoutCapacity",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Average",
}
_INVOCATIONS_PER_INSTANCE = {
    "MetricName": "InvocationsPerInstance",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Sum",
}

# Default tuning — callers can override via kwargs (typically pulled from workbench_meta).
_DEFAULT_MAX_CAPACITY = 4
_DEFAULT_ASYNC_TARGET = 2.0      # backlog per instance
_DEFAULT_REALTIME_TARGET = 750.0  # invocations per instance
_DEFAULT_SCALE_IN_COOLDOWN = 300
_DEFAULT_SCALE_OUT_COOLDOWN = 60
_DEFAULT_STEP_COOLDOWN = 60       # seconds between 0→1 step policy firings

_SERVICE_NS = "sagemaker"
_SCALABLE_DIM = "sagemaker:variant:DesiredInstanceCount"


def register_autoscaling(
    boto3_session,
    endpoint_name: str,
    min_capacity: int = 0,
    max_capacity: int = _DEFAULT_MAX_CAPACITY,
    target_value: Optional[float] = None,
    scale_in_cooldown: int = _DEFAULT_SCALE_IN_COOLDOWN,
    scale_out_cooldown: int = _DEFAULT_SCALE_OUT_COOLDOWN,
    raise_on_error: bool = True,
) -> dict:
    """Register auto-scaling for a SageMaker endpoint.

    Selects policy shape based on ``min_capacity``:
      * ``min_capacity == 0`` → async scale-to-zero (step + target-tracking)
      * ``min_capacity >= 1`` → realtime (target-tracking only)

    Args:
        boto3_session: Boto3 session for API calls.
        endpoint_name: Name of the SageMaker endpoint.
        min_capacity: Minimum instance count. 0 enables scale-to-zero.
        max_capacity: Maximum instance count under load.
        target_value: Target value for the target-tracking metric. Defaults to
            2.0 (backlog per instance) for async, 750.0 (invocations per
            instance) for realtime.
        scale_in_cooldown: Seconds to wait before scaling in.
        scale_out_cooldown: Seconds to wait before scaling out.
        raise_on_error: If True (default), re-raise exceptions after logging.
            Pass False to preserve previous best-effort behavior.

    Returns:
        Dict with 'scalable_targets' and 'scaling_policies' as reported by
        describe_* calls after registration, for verification / logging.
    """
    aas = boto3_session.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    is_async = min_capacity == 0

    try:
        # Register the scalable target. Idempotent — safe to call repeatedly.
        aas.register_scalable_target(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

        if is_async:
            # 1→N (and N→0) via target tracking on backlog per instance
            _put_target_tracking(
                aas,
                endpoint_name,
                resource_id,
                metric=_BACKLOG_PER_INSTANCE,
                target_value=target_value if target_value is not None else _DEFAULT_ASYNC_TARGET,
                scale_in_cooldown=scale_in_cooldown,
                scale_out_cooldown=scale_out_cooldown,
                policy_suffix="backlog-target",
            )
            # 0→1 via step scaling on the binary HasBacklogWithoutCapacity metric.
            cw = boto3_session.client("cloudwatch")
            _put_step_scaling_zero_to_one(aas, cw, endpoint_name, resource_id)
        else:
            _put_target_tracking(
                aas,
                endpoint_name,
                resource_id,
                metric=_INVOCATIONS_PER_INSTANCE,
                target_value=target_value if target_value is not None else _DEFAULT_REALTIME_TARGET,
                scale_in_cooldown=scale_in_cooldown,
                scale_out_cooldown=scale_out_cooldown,
                policy_suffix="invocations-target",
            )

    except Exception:
        log.exception(f"Failed to register auto-scaling for '{endpoint_name}'")
        if raise_on_error:
            raise
        return {"scalable_targets": [], "scaling_policies": []}

    # Post-registration verification — confirm what actually landed.
    verified = _describe_registration(aas, resource_id)
    targets = verified.get("scalable_targets", [])
    policies = verified.get("scaling_policies", [])
    log.important(
        f"Auto-scaling registered for '{endpoint_name}': "
        f"min={min_capacity}, max={max_capacity}, "
        f"mode={'async' if is_async else 'realtime'}, "
        f"scalable_targets={len(targets)}, scaling_policies={len(policies)}"
    )
    for pol in policies:
        log.info(f"  policy: {pol.get('PolicyName')} type={pol.get('PolicyType')}")
    return verified


def deregister_autoscaling(boto3_session, endpoint_name: str, raise_on_error: bool = False) -> None:
    """Remove all scaling policies and the scalable target for an endpoint.

    Idempotent: a missing scalable target is a quiet no-op (a previously failed
    deploy may have left nothing to clean up). Other unexpected errors are
    logged with full traceback and re-raised when ``raise_on_error=True``.
    """
    aas = boto3_session.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Delete all scaling policies attached to the scalable target first.
    try:
        policies = aas.describe_scaling_policies(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
        ).get("ScalingPolicies", [])
    except Exception:
        log.exception(f"Failed to list scaling policies for '{endpoint_name}'")
        if raise_on_error:
            raise
        return

    for pol in policies:
        try:
            aas.delete_scaling_policy(
                PolicyName=pol["PolicyName"],
                ServiceNamespace=_SERVICE_NS,
                ResourceId=resource_id,
                ScalableDimension=_SCALABLE_DIM,
            )
        except aas.exceptions.ObjectNotFoundException:
            pass  # Already gone — fine.
        except Exception:
            log.exception(f"Failed to delete scaling policy '{pol['PolicyName']}'")
            if raise_on_error:
                raise

    # Then deregister the scalable target itself.
    try:
        aas.deregister_scalable_target(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
        )
        log.important(f"Auto-scaling deregistered for '{endpoint_name}'")
    except aas.exceptions.ObjectNotFoundException:
        # Nothing to deregister — likely a previously failed deploy never got this far.
        log.info(f"No scalable target found for '{endpoint_name}' — nothing to deregister")
    except Exception:
        log.exception(f"Failed to deregister scalable target for '{endpoint_name}'")
        if raise_on_error:
            raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _put_target_tracking(
    aas,
    endpoint_name: str,
    resource_id: str,
    metric: dict,
    target_value: float,
    scale_in_cooldown: int,
    scale_out_cooldown: int,
    policy_suffix: str,
) -> None:
    """Register a TargetTrackingScaling policy on a custom metric.

    FUTURE NOTE: target-tracking auto-creates two CloudWatch alarms with
    UUID-suffixed, unreadable names like::

        TargetTracking-endpoint/<name>/variant/AllTraffic-AlarmHigh-<uuid>
        TargetTracking-endpoint/<name>/variant/AllTraffic-AlarmLow-<uuid>

    AWS owns those names; renaming them breaks the policy's alarm lookup.
    If the UUID noise becomes a real problem, replace this target-tracking
    policy with two explicit StepScaling policies (high/low) on the same
    metric — that gives us full alarm-naming control at the cost of writing
    the scale-up/down math ourselves (step sizes, scale-in thresholds).
    For now we accept AWS's naming convention.
    """
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-{policy_suffix}",
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
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


def _put_step_scaling_zero_to_one(aas, cw, endpoint_name: str, resource_id: str) -> None:
    """Register a StepScaling policy that takes the endpoint from 0 → 1 instance.

    Uses the ``HasBacklogWithoutCapacity`` binary metric (1 when there is
    backlog and no capacity, 0 otherwise). When >= 1, add one instance.
    This is the AWS-recommended pattern for async scale-from-zero.

    StepScaling policies don't auto-create their alarm (unlike target-tracking
    which does), so we create the alarm ourselves and wire it to the policy ARN.

    Args:
        aas: application-autoscaling boto3 client
        cw: cloudwatch boto3 client
        endpoint_name: SageMaker endpoint name
        resource_id: "endpoint/<name>/variant/AllTraffic"
    """
    policy_name = f"{endpoint_name}-zero-to-one"
    pol_resp = aas.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "Cooldown": _DEFAULT_STEP_COOLDOWN,
            "MetricAggregationType": "Average",
            "StepAdjustments": [
                {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1},
            ],
        },
    )
    policy_arn = pol_resp["PolicyARN"]

    cw.put_metric_alarm(
        AlarmName=f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-has-backlog",
        MetricName=_HAS_BACKLOG_WITHOUT_CAPACITY["MetricName"],
        Namespace=_HAS_BACKLOG_WITHOUT_CAPACITY["Namespace"],
        Statistic=_HAS_BACKLOG_WITHOUT_CAPACITY["Statistic"],
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Period=60,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1.0,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="missing",
        AlarmActions=[policy_arn],
    )


def _describe_registration(aas, resource_id: str) -> dict:
    """Describe scalable targets and policies for verification / logging."""
    try:
        targets = aas.describe_scalable_targets(
            ServiceNamespace=_SERVICE_NS,
            ResourceIds=[resource_id],
            ScalableDimension=_SCALABLE_DIM,
        ).get("ScalableTargets", [])
        policies = aas.describe_scaling_policies(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
        ).get("ScalingPolicies", [])
        return {"scalable_targets": targets, "scaling_policies": policies}
    except Exception:
        log.exception("Failed to describe auto-scaling registration")
        return {"scalable_targets": [], "scaling_policies": []}
