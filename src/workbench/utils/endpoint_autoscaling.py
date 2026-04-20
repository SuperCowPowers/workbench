"""Auto-scaling for SageMaker endpoints (batch-async and realtime).

Architecture
------------
Scaling is modeled as a **mode** — a named strategy that emits a list of
policy specs. The mode shapes the endpoint's behaviour:

    "batch"      → async endpoints. Step policies:
                     - scale-out on HasBacklogWithoutCapacity >= 1
                       ExactCapacity = max_capacity  (instant jump to full fleet)
                     - scale-in on ApproximateBacklogSize < 1 for N minutes
                       ExactCapacity = min_capacity  (drop to zero fast)
                   Optimized for "pile of work arrives, chew through it, go
                   cold." Predictable, no target-tracking to fight the jumps.

    "realtime"   → realtime endpoints. Target-tracking on InvocationsPerInstance.

    "dynamic"    → (future) async with target-tracking on backlog-per-instance
                   for continuous / unpredictable traffic.

Extending with a new mode is a single-function change: define
``_<mode>_mode_specs(...)`` that returns a list of ``StepPolicySpec`` /
``TargetTrackingSpec``, then add it to ``_MODE_HANDLERS``. Generic
installers translate specs → AWS API calls.

Naming convention
-----------------
Scaling policies and manually-created CloudWatch alarms are named
``{endpoint_name}-{role}`` (e.g. ``smiles-to-3d-full-v1-scale-out``).
``deregister_autoscaling`` finds them by prefix, so we're not hard-coding
a name list that drifts when modes evolve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

log = logging.getLogger("workbench")

# ---------------------------------------------------------------------------
# AWS metric definitions
# ---------------------------------------------------------------------------
_HAS_BACKLOG_WITHOUT_CAPACITY = {
    "MetricName": "HasBacklogWithoutCapacity",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Average",
}
_APPROXIMATE_BACKLOG_SIZE = {
    "MetricName": "ApproximateBacklogSize",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Maximum",
}
_INVOCATIONS_PER_INSTANCE = {
    "MetricName": "InvocationsPerInstance",
    "Namespace": "AWS/SageMaker",
    "Statistic": "Sum",
}

# ---------------------------------------------------------------------------
# Defaults — callers override via kwargs (typically sourced from workbench_meta).
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CAPACITY = 8
_DEFAULT_SCALE_IN_IDLE_MINUTES = 15  # batch mode only
_DEFAULT_REALTIME_TARGET = 750.0  # invocations per instance

_SERVICE_NS = "sagemaker"
_SCALABLE_DIM = "sagemaker:variant:DesiredInstanceCount"


# ---------------------------------------------------------------------------
# Policy specs (data-only — installation logic is elsewhere)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepPolicySpec:
    """A step-scaling policy + its triggering CloudWatch alarm.

    ``adjustment_type="ExactCapacity"`` is preferred for batch mode because
    it's idempotent — an alarm re-firing won't overshoot. ``ChangeInCapacity``
    is available for future modes that want additive semantics.
    """

    role: str  # "scale-out" | "scale-in" — suffix for policy & alarm names
    metric: dict
    threshold: float
    comparison: str  # "GreaterThanOrEqualToThreshold" | "LessThanThreshold" | ...
    evaluation_periods: int
    datapoints_to_alarm: int
    period_seconds: int
    treat_missing_data: str
    adjustment_type: str  # "ExactCapacity" | "ChangeInCapacity"
    adjustment: int
    cooldown_seconds: int


@dataclass(frozen=True)
class TargetTrackingSpec:
    """A target-tracking scaling policy on a custom metric.

    AWS auto-creates the high/low alarms (with UUID-suffixed names we don't
    control); they're auto-cleaned when the policy is deleted.
    """

    role: str  # e.g. "invocations-target" | "backlog-target"
    metric: dict
    target_value: float
    scale_in_cooldown: int
    scale_out_cooldown: int


Spec = Union[StepPolicySpec, TargetTrackingSpec]


# ---------------------------------------------------------------------------
# Mode handlers — each returns a list of policy specs for the given params
# ---------------------------------------------------------------------------
def _batch_mode_specs(max_capacity: int, min_capacity: int, scale_in_idle_minutes: int) -> List[Spec]:
    """Specs for batch-async endpoints. Jump 0→max, idle→min, no mid-range dithering."""
    return [
        # 0 → max_capacity when backlog arrives and there's no capacity.
        # Fires within 60s of the first queued request on a cold endpoint.
        StepPolicySpec(
            role="scale-out",
            metric=_HAS_BACKLOG_WITHOUT_CAPACITY,
            threshold=1.0,
            comparison="GreaterThanOrEqualToThreshold",
            evaluation_periods=1,
            datapoints_to_alarm=1,
            period_seconds=60,
            treat_missing_data="missing",
            adjustment_type="ExactCapacity",
            adjustment=max_capacity,
            cooldown_seconds=60,
        ),
        # max_capacity → min_capacity when the queue has been empty for a
        # sustained window. `ApproximateBacklogSize < 1` with N consecutive
        # 60-s datapoints means N minutes of zero queue. SageMaker won't kill
        # an instance mid-invocation anyway, so this is a "truly idle" signal.
        StepPolicySpec(
            role="scale-in",
            metric=_APPROXIMATE_BACKLOG_SIZE,
            threshold=1.0,
            comparison="LessThanThreshold",
            evaluation_periods=scale_in_idle_minutes,
            datapoints_to_alarm=scale_in_idle_minutes,
            period_seconds=60,
            treat_missing_data="notBreaching",
            adjustment_type="ExactCapacity",
            adjustment=min_capacity,
            cooldown_seconds=60,
        ),
    ]


def _realtime_mode_specs(target_value: float) -> List[Spec]:
    """Specs for realtime endpoints — single target-tracking policy."""
    return [
        TargetTrackingSpec(
            role="invocations-target",
            metric=_INVOCATIONS_PER_INSTANCE,
            target_value=target_value,
            scale_in_cooldown=900,
            scale_out_cooldown=60,
        ),
    ]


_MODE_HANDLERS = {
    "batch": _batch_mode_specs,
    "realtime": _realtime_mode_specs,
    # Future: "dynamic": _dynamic_mode_specs,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_autoscaling(
    boto3_session,
    endpoint_name: str,
    auto_scaling_mode: str = "batch",
    min_capacity: int = 0,
    max_capacity: int = _DEFAULT_MAX_CAPACITY,
    scale_in_idle_minutes: int = _DEFAULT_SCALE_IN_IDLE_MINUTES,
    realtime_target: float = _DEFAULT_REALTIME_TARGET,
    raise_on_error: bool = True,
) -> dict:
    """Register autoscaling policies for a SageMaker endpoint.

    Args:
        boto3_session: Active boto3 Session.
        endpoint_name: SageMaker endpoint name.
        auto_scaling_mode: Scaling strategy. One of ``"batch"`` (async step-scaling) or
            ``"realtime"`` (invocations target-tracking). Default ``"batch"``.
        min_capacity: Autoscaler floor. Default 0 (scale-to-zero). Batch mode's
            scale-in action drops to this value; realtime should usually be >= 1.
        max_capacity: Autoscaler ceiling. Default 8.
        scale_in_idle_minutes: Batch mode only. Minutes of empty queue before
            scaling in. Default 15.
        realtime_target: Realtime mode only. Target InvocationsPerInstance for
            the target-tracking policy. Default 750.
        raise_on_error: If True, re-raise exceptions after logging.

    Returns:
        Dict with 'scalable_targets' and 'scaling_policies' from post-registration
        describe calls.
    """
    if auto_scaling_mode not in _MODE_HANDLERS:
        raise ValueError(
            f"Unsupported auto_scaling_mode {auto_scaling_mode!r}. Valid: {sorted(_MODE_HANDLERS)}"
        )

    aas = boto3_session.client("application-autoscaling")
    cw = boto3_session.client("cloudwatch")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Build mode-specific specs. Each mode owns exactly the params it needs.
    if auto_scaling_mode == "batch":
        specs = _batch_mode_specs(max_capacity, min_capacity, scale_in_idle_minutes)
    elif auto_scaling_mode == "realtime":
        specs = _realtime_mode_specs(realtime_target)
    else:  # defensive — _MODE_HANDLERS check above should prevent this
        raise ValueError(f"Unsupported auto_scaling_mode {auto_scaling_mode!r}")

    try:
        # Register the scalable target (idempotent). Min/max bound everything else.
        aas.register_scalable_target(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

        # Install each spec. Dispatch by type keeps add-a-new-spec-kind simple.
        for spec in specs:
            if isinstance(spec, StepPolicySpec):
                _install_step_policy(aas, cw, endpoint_name, resource_id, spec)
            elif isinstance(spec, TargetTrackingSpec):
                _install_target_tracking(aas, endpoint_name, resource_id, spec)
            else:  # defensive
                raise TypeError(f"Unknown spec type: {type(spec).__name__}")

    except Exception:
        log.exception(
            f"Failed to register autoscaling for '{endpoint_name}' "
            f"(auto_scaling_mode={auto_scaling_mode})"
        )
        if raise_on_error:
            raise
        return {"scalable_targets": [], "scaling_policies": []}

    verified = _describe_registration(aas, resource_id)
    log.important(
        f"Autoscaling registered: endpoint='{endpoint_name}' "
        f"auto_scaling_mode={auto_scaling_mode} "
        f"min={min_capacity} max={max_capacity} "
        f"policies={len(verified['scaling_policies'])} "
        f"specs={[s.role for s in specs]}"
    )
    for spec in specs:
        _log_spec(endpoint_name, spec)
    return verified


def deregister_autoscaling(boto3_session, endpoint_name: str, raise_on_error: bool = False) -> None:
    """Remove all scaling policies, the scalable target, and our manually-created
    CloudWatch alarms for an endpoint.

    Discovery-based: we don't keep a hardcoded list of policy/alarm names, so
    this cleans up regardless of which mode the endpoint was deployed with
    (and across mode changes over the endpoint's lifetime).
    """
    aas = boto3_session.client("application-autoscaling")
    cw = boto3_session.client("cloudwatch")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # 1. Delete every scaling policy attached to this scalable target.
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
            pass  # already gone
        except Exception:
            log.exception(f"Failed to delete scaling policy '{pol['PolicyName']}'")
            if raise_on_error:
                raise

    # 2. Deregister the scalable target itself.
    try:
        aas.deregister_scalable_target(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
        )
        log.important(f"Autoscaling deregistered for '{endpoint_name}'")
    except aas.exceptions.ObjectNotFoundException:
        log.info(f"No scalable target found for '{endpoint_name}' — nothing to deregister")
    except Exception:
        log.exception(f"Failed to deregister scalable target for '{endpoint_name}'")
        if raise_on_error:
            raise

    # 3. Clean up our step-scaling alarms by prefix. Target-tracking alarms
    #    are auto-deleted by AWS when the policy is deleted above, so we only
    #    need to handle ones we put_metric_alarm'd ourselves.
    _delete_alarms_by_prefix(cw, endpoint_name, raise_on_error=raise_on_error)


# ---------------------------------------------------------------------------
# Installers — translate specs → AWS API calls
# ---------------------------------------------------------------------------
def _install_step_policy(aas, cw, endpoint_name: str, resource_id: str, spec: StepPolicySpec) -> None:
    """Install a StepScaling policy + the CloudWatch alarm that fires it."""
    policy_name = _policy_name(endpoint_name, spec.role)
    alarm_name = _alarm_name(endpoint_name, spec.role)

    pol_resp = aas.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": spec.adjustment_type,
            "Cooldown": spec.cooldown_seconds,
            "MetricAggregationType": "Average",
            "StepAdjustments": [
                # MetricIntervalLowerBound=0 means "at or above the alarm's threshold."
                {"MetricIntervalLowerBound": 0, "ScalingAdjustment": spec.adjustment},
            ],
        },
    )

    cw.put_metric_alarm(
        AlarmName=alarm_name,
        MetricName=spec.metric["MetricName"],
        Namespace=spec.metric["Namespace"],
        Statistic=spec.metric["Statistic"],
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Period=spec.period_seconds,
        EvaluationPeriods=spec.evaluation_periods,
        DatapointsToAlarm=spec.datapoints_to_alarm,
        Threshold=spec.threshold,
        ComparisonOperator=spec.comparison,
        TreatMissingData=spec.treat_missing_data,
        AlarmActions=[pol_resp["PolicyARN"]],
    )


def _install_target_tracking(aas, endpoint_name: str, resource_id: str, spec: TargetTrackingSpec) -> None:
    """Install a TargetTrackingScaling policy. AWS auto-creates the alarms."""
    aas.put_scaling_policy(
        PolicyName=_policy_name(endpoint_name, spec.role),
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": spec.target_value,
            "CustomizedMetricSpecification": {
                **spec.metric,
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
            },
            "ScaleInCooldown": spec.scale_in_cooldown,
            "ScaleOutCooldown": spec.scale_out_cooldown,
        },
    )


# ---------------------------------------------------------------------------
# Naming conventions (one source of truth — no drift between install & cleanup)
# ---------------------------------------------------------------------------
def _policy_name(endpoint_name: str, role: str) -> str:
    return f"{endpoint_name}-{role}"


def _alarm_name(endpoint_name: str, role: str) -> str:
    return f"{endpoint_name}-{role}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _delete_alarms_by_prefix(cw, endpoint_name: str, raise_on_error: bool = False) -> None:
    """Delete all CloudWatch alarms whose name starts with the endpoint name.

    Also sweeps legacy alarm names (pre-refactor) so cleanup works across the
    rename. Legacy names: ``TargetTracking-endpoint/{name}/variant/AllTraffic-has-backlog``
    and ``...-rapid-scale-out``.
    """
    try:
        # Current convention: alarms prefixed with the endpoint name.
        matching: List[str] = []
        paginator = cw.get_paginator("describe_alarms")
        for page in paginator.paginate(AlarmNamePrefix=f"{endpoint_name}-"):
            matching.extend(a["AlarmName"] for a in page.get("MetricAlarms", []))

        # Legacy names from the pre-refactor code. Harmless if they don't exist —
        # delete_alarms ignores missing names silently.
        legacy = [
            f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-has-backlog",
            f"TargetTracking-endpoint/{endpoint_name}/variant/AllTraffic-rapid-scale-out",
        ]

        all_alarms = list(dict.fromkeys(matching + legacy))  # dedupe, preserve order
        if all_alarms:
            # delete_alarms accepts up to 100 names per call.
            for batch in _chunked(all_alarms, 100):
                cw.delete_alarms(AlarmNames=list(batch))
    except Exception:
        log.exception(f"Failed to delete CloudWatch alarms for '{endpoint_name}'")
        if raise_on_error:
            raise


def _chunked(seq: Iterable, size: int) -> Iterable[list]:
    buf: list = []
    for item in seq:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _describe_registration(aas, resource_id: str) -> dict:
    """Describe scalable targets and policies for post-registration verification."""
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
        log.exception("Failed to describe autoscaling registration")
        return {"scalable_targets": [], "scaling_policies": []}


_COMPARISON_SYMBOLS = {
    "GreaterThanOrEqualToThreshold": ">=",
    "GreaterThanThreshold": ">",
    "LessThanOrEqualToThreshold": "<=",
    "LessThanThreshold": "<",
}


def _log_spec(endpoint_name: str, spec: Spec) -> None:
    """One structured line per policy: trigger → effect. Useful when debugging
    'why did (or didn't) this endpoint scale'."""
    if isinstance(spec, StepPolicySpec):
        window_s = spec.evaluation_periods * spec.period_seconds
        cmp_sym = _COMPARISON_SYMBOLS.get(spec.comparison, spec.comparison)
        log.info(
            f"  [{spec.role}] {spec.metric['MetricName']} {cmp_sym} {spec.threshold} "
            f"for {window_s}s → {spec.adjustment_type}={spec.adjustment}"
        )
    elif isinstance(spec, TargetTrackingSpec):
        log.info(
            f"  [{spec.role}] target {spec.metric['MetricName']}={spec.target_value} "
            f"(scale-in cooldown {spec.scale_in_cooldown}s, scale-out {spec.scale_out_cooldown}s)"
        )
