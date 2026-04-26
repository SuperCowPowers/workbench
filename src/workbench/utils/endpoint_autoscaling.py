"""Auto-scaling for SageMaker endpoints.

Two modes:
  - "batch": step policies for async endpoints. ApproximateBacklogSize >= 1
    jumps to max in one step; < 1 for N minutes drains to min. Pile of work
    arrives, chew through it, go cold.
  - "realtime": single target-tracking policy on InvocationsPerInstance.

Resources are named ``{endpoint_name}-{role}`` (e.g. ``ep-scale-out``).
``deregister_autoscaling`` finds them by prefix.
"""

from __future__ import annotations

import logging
from typing import Optional

from botocore.exceptions import ClientError

log = logging.getLogger("workbench")

_SERVICE_NS = "sagemaker"
_SCALABLE_DIM = "sagemaker:variant:DesiredInstanceCount"

_DEFAULT_MAX_CAPACITY = 8
_DEFAULT_SCALE_IN_IDLE_MINUTES = 5
_DEFAULT_REALTIME_TARGET = 750.0


def register_autoscaling(
    boto3_session,
    endpoint_name: str,
    auto_scaling_mode: str = "batch",
    min_capacity: int = 0,
    max_capacity: int = _DEFAULT_MAX_CAPACITY,
    scale_in_idle_minutes: int = _DEFAULT_SCALE_IN_IDLE_MINUTES,
    realtime_target: float = _DEFAULT_REALTIME_TARGET,
    raise_on_error: bool = True,
) -> None:
    """Register autoscaling policies for a SageMaker endpoint.

    Args:
        boto3_session: Active boto3 Session.
        endpoint_name: SageMaker endpoint name.
        auto_scaling_mode: ``"batch"`` (async step-scaling) or ``"realtime"``
            (invocations target-tracking). Default ``"batch"``.
        min_capacity: Autoscaler floor. Default 0 (scale-to-zero). Realtime
            should usually be >= 1.
        max_capacity: Autoscaler ceiling. Default 8.
        scale_in_idle_minutes: Batch only. Minutes of empty queue before
            scaling in. Default 5.
        realtime_target: Realtime only. Target InvocationsPerInstance. Default 750.
        raise_on_error: If True, re-raise after logging.
    """
    if auto_scaling_mode not in ("batch", "realtime"):
        raise ValueError(f"Unsupported auto_scaling_mode {auto_scaling_mode!r}. Valid: ['batch', 'realtime']")

    aas = boto3_session.client("application-autoscaling")
    cw = boto3_session.client("cloudwatch")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    _preflight_quota_check(boto3_session, endpoint_name, max_capacity)

    try:
        aas.register_scalable_target(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )
        if auto_scaling_mode == "batch":
            _install_batch(aas, cw, endpoint_name, resource_id, min_capacity, max_capacity, scale_in_idle_minutes)
        else:
            _install_realtime(aas, endpoint_name, resource_id, realtime_target)
    except Exception:
        log.exception(f"Failed to register autoscaling for '{endpoint_name}' (mode={auto_scaling_mode})")
        if raise_on_error:
            raise
        return

    log.important(
        f"Autoscaling registered: endpoint='{endpoint_name}' mode={auto_scaling_mode} "
        f"min={min_capacity} max={max_capacity}"
    )


def deregister_autoscaling(boto3_session, endpoint_name: str) -> None:
    """Best-effort cleanup of all scaling policies, the scalable target, and our
    CloudWatch alarms for an endpoint. Errors are logged, never raised.

    Discovery-based: works regardless of which mode the endpoint was deployed
    with (and across mode changes over its lifetime).
    """
    aas = boto3_session.client("application-autoscaling")
    cw = boto3_session.client("cloudwatch")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    try:
        policies = aas.describe_scaling_policies(
            ServiceNamespace=_SERVICE_NS,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIM,
        ).get("ScalingPolicies", [])
        for pol in policies:
            try:
                aas.delete_scaling_policy(
                    PolicyName=pol["PolicyName"],
                    ServiceNamespace=_SERVICE_NS,
                    ResourceId=resource_id,
                    ScalableDimension=_SCALABLE_DIM,
                )
            except aas.exceptions.ObjectNotFoundException:
                pass

        try:
            aas.deregister_scalable_target(
                ServiceNamespace=_SERVICE_NS,
                ResourceId=resource_id,
                ScalableDimension=_SCALABLE_DIM,
            )
            log.important(f"Autoscaling deregistered for '{endpoint_name}'")
        except aas.exceptions.ObjectNotFoundException:
            log.info(f"No scalable target for '{endpoint_name}' — nothing to deregister")

        # Sweep our endpoint-prefixed CloudWatch alarms. Target-tracking alarms
        # have UUID names (no endpoint prefix) and are auto-deleted by AWS when
        # their policy is deleted above.
        alarms = []
        for page in cw.get_paginator("describe_alarms").paginate(AlarmNamePrefix=f"{endpoint_name}-"):
            alarms.extend(a["AlarmName"] for a in page.get("MetricAlarms", []))
        if alarms:
            cw.delete_alarms(AlarmNames=alarms)
    except Exception:
        log.exception(f"Failed to deregister autoscaling for '{endpoint_name}'")


def _install_batch(aas, cw, endpoint_name, resource_id, min_capacity, max_capacity, scale_in_idle_minutes):
    """Step policies for async/batch endpoints.

    Both triggers key off ApproximateBacklogSize so they're symmetric and work
    at any current instance count. (HasBacklogWithoutCapacity only fires when
    instances=0, so it can't take us N→max once instances are running.)
    """
    _put_step_policy_with_alarm(
        aas,
        cw,
        endpoint_name,
        resource_id,
        role="scale-out",
        adjustment=max_capacity,
        comparison="GreaterThanOrEqualToThreshold",
        evaluation_periods=1,
    )
    _put_step_policy_with_alarm(
        aas,
        cw,
        endpoint_name,
        resource_id,
        role="scale-in",
        adjustment=min_capacity,
        comparison="LessThanThreshold",
        evaluation_periods=scale_in_idle_minutes,
    )


def _put_step_policy_with_alarm(aas, cw, endpoint_name, resource_id, role, adjustment, comparison, evaluation_periods):
    """Install one ExactCapacity step policy + its CloudWatch alarm.

    Step bound is signed against the alarm threshold: GreaterThan* alarms use
    LowerBound=0 (positive breach), LessThan* alarms use UpperBound=0 (negative
    breach). Picking the wrong bound silently no-ops — the alarm fires but no
    step matches the breach.
    """
    name = f"{endpoint_name}-{role}"
    bound = "MetricIntervalUpperBound" if comparison.startswith("Less") else "MetricIntervalLowerBound"

    pol = aas.put_scaling_policy(
        PolicyName=name,
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "Cooldown": 0,
            "MetricAggregationType": "Average",
            "StepAdjustments": [{bound: 0, "ScalingAdjustment": adjustment}],
        },
    )

    cw.put_metric_alarm(
        AlarmName=name,
        MetricName="ApproximateBacklogSize",
        Namespace="AWS/SageMaker",
        Statistic="Maximum",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Period=60,
        EvaluationPeriods=evaluation_periods,
        DatapointsToAlarm=evaluation_periods,
        Threshold=1.0,
        ComparisonOperator=comparison,
        TreatMissingData="notBreaching",
        AlarmActions=[pol["PolicyARN"]],
    )


def _install_realtime(aas, endpoint_name, resource_id, target_value):
    """Target-tracking on InvocationsPerInstance. AWS auto-creates the alarms."""
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-invocations-target",
        ServiceNamespace=_SERVICE_NS,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIM,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": target_value,
            "CustomizedMetricSpecification": {
                "MetricName": "InvocationsPerInstance",
                "Namespace": "AWS/SageMaker",
                "Statistic": "Sum",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
            },
            "ScaleInCooldown": 900,
            "ScaleOutCooldown": 60,
        },
    )


def _preflight_quota_check(boto3_session, endpoint_name: str, max_capacity: int) -> None:
    """Warn if max_capacity exceeds the account's per-instance-type endpoint quota.

    Purely advisory — never raises. Fails open on permissions, serverless,
    quota-not-found, or unknown region.

    Requires ``servicequotas:ListServiceQuotas`` IAM permission. Without it,
    the preflight silently passes.
    """
    try:
        sm = boto3_session.client("sagemaker")
        ep = sm.describe_endpoint(EndpointName=endpoint_name)
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        variants = cfg.get("ProductionVariants", [])
        if not variants:
            return
        instance_type = variants[0].get("InstanceType")
        if not instance_type:  # serverless
            return

        target_name = f"{instance_type} for endpoint usage"
        sq = boto3_session.client("service-quotas")
        quota: Optional[int] = None
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page.get("Quotas", []):
                if q.get("QuotaName") == target_name:
                    quota = int(q["Value"])
                    break
            if quota is not None:
                break

        if quota is None:
            return
        if quota < max_capacity:
            log.warning(
                f"⚠ Autoscaling preflight: max_capacity={max_capacity} exceeds account quota "
                f"'{target_name}' = {quota}. Scale-out will be capped at {quota} instances. "
                f"Request increase: AWS Console → Service Quotas → SageMaker → '{instance_type}'."
            )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDeniedException", "AccessDenied", "UnauthorizedOperation"):
            log.info(
                "Autoscaling quota preflight skipped — execution role lacks "
                "'servicequotas:ListServiceQuotas'. Add it to enable the warning."
            )
        else:
            log.debug(f"Quota preflight skipped for '{endpoint_name}' ({code})", exc_info=True)
    except Exception:
        log.debug(f"Quota preflight skipped for '{endpoint_name}'", exc_info=True)
