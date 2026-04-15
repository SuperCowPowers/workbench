"""Auto-scaling utilities for SageMaker endpoints (async and realtime).

Async endpoints use THREE policies because each covers a transition the others
structurally can't:

    StepScaling    on HasBacklogWithoutCapacity            → 0 → 1  (fast)
    StepScaling    on ApproximateBacklogSizePerInstance≥5  → 1 → N  (fast)
    TargetTracking on ApproximateBacklogSizePerInstance    → scale-in + fine-tune

The first step policy is required because target-tracking on per-instance
metrics hits divide-by-zero at 0 instances. The second exists because
target-tracking's 3-min alarm evaluation + internal deliberation made real
scale-out take ~25 min for bursty batch loads — the step policy fires in ~1 min.

Realtime endpoints use a single target-tracking policy on ``InvocationsPerInstance``.

FUTURE NOTE: target-tracking here creates CloudWatch alarms with unreadable
UUID-suffixed names (``...-AlarmHigh-<uuid>``, ``...-AlarmLow-<uuid>``). If
those names become a real problem — e.g., the dashboard/alerting surfaces them
and users get confused — we can swap target-tracking for a third step policy
(scale-down on ``per_instance < 0.5 for 15min``). Same 3-policy count, all
human-named alarms, consistent mental model. Tradeoff: we write the scale-in
logic ourselves and lose AWS's built-in smoothing for bursty loads. Not worth
doing today for pure "big batch, single call" workloads, but reconsider if the
endpoint ever sees mixed/variable traffic patterns.
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
# max_capacity=8 gives headroom for bursty async batches; scale-to-zero keeps idle cost at 0.
_DEFAULT_MAX_CAPACITY = 8
_DEFAULT_ASYNC_TARGET = 2.0  # backlog per instance
_DEFAULT_REALTIME_TARGET = 750.0  # invocations per instance
_DEFAULT_SCALE_IN_COOLDOWN = 300
_DEFAULT_SCALE_OUT_COOLDOWN = 60
_DEFAULT_STEP_COOLDOWN = 60  # seconds between 0→1 step policy firings

# Rapid scale-out step policy thresholds (async only). Fires within ~1-2 min of
# backlog spiking — much faster than target-tracking's ~5-min alarm evaluation.
# Target-tracking still runs in parallel for steady-state adjustment and scale-in.
_RAPID_SCALE_OUT_THRESHOLD = 5.0       # backlog per instance
_RAPID_SCALE_OUT_MINOR_STEP = 1        # backlog/instance in [5, 10) → add 1
_RAPID_SCALE_OUT_MAJOR_STEP = 3        # backlog/instance in [10, ∞)  → add 3
_RAPID_SCALE_OUT_COOLDOWN = 120        # don't re-fire within 2 min of acting

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
            cw = boto3_session.client("cloudwatch")

            # 1→N (and N→0) via target tracking on backlog per instance.
            # Good for steady-state and scale-in; conservative on scale-out.
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
            _put_step_scaling_zero_to_one(aas, cw, endpoint_name, resource_id)
            # Rapid 1→N scale-out via step scaling on per-instance backlog.
            # Fires in ~1 min (vs target-tracking's ~5 min) so big batches get
            # instances in parallel quickly instead of crawling on 1 instance.
            _put_step_scaling_rapid_scale_out(aas, cw, endpoint_name, resource_id)
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

    # Clean up manually-created CloudWatch alarms. Target-tracking auto-removes
    # its own alarms on policy deletion, but step-scaling alarms are ones we
    # put_metric_alarm'd ourselves and must delete ourselves.
    cw = boto3_session.client("cloudwatch")
    step_alarm_names = [
        f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-has-backlog",
        f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-rapid-scale-out",
    ]
    try:
        cw.delete_alarms(AlarmNames=step_alarm_names)
    except Exception:
        # DeleteAlarms is idempotent — missing alarms are silently ignored by AWS —
        # so a failure here is a real error (e.g., permissions).
        log.exception(f"Failed to delete step-scaling alarms for '{endpoint_name}'")
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


def _put_step_scaling_rapid_scale_out(aas, cw, endpoint_name: str, resource_id: str) -> None:
    """Register a StepScaling policy for aggressive 1→N scale-out.

    Fires on ``ApproximateBacklogSizePerInstance`` with a 1-minute evaluation
    period (vs target-tracking's 3-min), so big async batches don't sit on
    one instance waiting for target-tracking to deliberate. Step ladder:

        backlog_per_instance in [threshold, threshold+5)   → add MINOR_STEP
        backlog_per_instance in [threshold+5, ∞)           → add MAJOR_STEP

    Target-tracking still runs in parallel — it handles steady-state fine-tuning
    and scale-in. This policy's job is just to get off 1 instance *fast* when
    there's a real pile-up.
    """
    policy_name = f"{endpoint_name}-rapid-scale-out"
    pol_resp = aas.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "Cooldown": _RAPID_SCALE_OUT_COOLDOWN,
            "MetricAggregationType": "Average",
            # MetricIntervalLowerBound/UpperBound are offsets from the alarm's
            # Threshold (= _RAPID_SCALE_OUT_THRESHOLD), so:
            #   [0, 5)  means [threshold, threshold+5)
            #   [5, ∞)  means [threshold+5, ∞)
            "StepAdjustments": [
                {
                    "MetricIntervalLowerBound": 0,
                    "MetricIntervalUpperBound": 5,
                    "ScalingAdjustment": _RAPID_SCALE_OUT_MINOR_STEP,
                },
                {
                    "MetricIntervalLowerBound": 5,
                    "ScalingAdjustment": _RAPID_SCALE_OUT_MAJOR_STEP,
                },
            ],
        },
    )
    policy_arn = pol_resp["PolicyARN"]

    cw.put_metric_alarm(
        AlarmName=f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-rapid-scale-out",
        MetricName=_BACKLOG_PER_INSTANCE["MetricName"],
        Namespace=_BACKLOG_PER_INSTANCE["Namespace"],
        Statistic=_BACKLOG_PER_INSTANCE["Statistic"],
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Period=60,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=_RAPID_SCALE_OUT_THRESHOLD,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
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
