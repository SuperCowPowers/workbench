"""Backfill description + tags on existing async-endpoint auto-scaling alarms.

New alarms get this metadata from ``workbench.utils.endpoint_autoscaling``, but
alarms deployed before that change need to be updated in place.

Identifies workbench-managed autoscaling alarms by:
  - MetricName == "ApproximateBacklogSize"
  - AlarmActions contains an application-autoscaling scaling-policy ARN
  - AlarmName ends with "-scale-out" or "-scale-in"

For each match:
  - re-puts the alarm with all existing config + AlarmDescription
    (put_metric_alarm has no partial-update API — must re-specify everything)
  - applies tags via tag_resource (Tags param on put_metric_alarm is
    creation-only and silently ignored on update)

Usage:
    python scripts/admin/backfill_autoscaling_alarm_metadata.py            # dry-run
    python scripts/admin/backfill_autoscaling_alarm_metadata.py --apply    # write
    python scripts/admin/backfill_autoscaling_alarm_metadata.py --apply --endpoint my-ep
"""

import argparse
import sys

from workbench.api import Meta

SCALE_ROLES = ("scale-out", "scale-in")


def _is_workbench_autoscaling_alarm(alarm: dict) -> tuple[bool, str | None]:
    """Return (is_match, role) where role is 'scale-out' or 'scale-in'."""
    if alarm.get("MetricName") != "ApproximateBacklogSize":
        return False, None
    if not any(":autoscaling:" in a for a in alarm.get("AlarmActions", [])):
        return False, None
    name = alarm["AlarmName"]
    for role in SCALE_ROLES:
        if name.endswith(f"-{role}"):
            return True, role
    return False, None


def _endpoint_name_from_alarm(alarm: dict) -> str | None:
    for dim in alarm.get("Dimensions", []):
        if dim.get("Name") == "EndpointName":
            return dim.get("Value")
    return None


def _description_for(endpoint_name: str, role: str) -> str:
    return (
        f"Workbench auto-scaling alarm ({role}) for SageMaker async endpoint "
        f"'{endpoint_name}'. Managed by workbench.utils.endpoint_autoscaling."
    )


def _tags_for(endpoint_name: str) -> list[dict]:
    return [
        {"Key": "ManagedBy", "Value": "workbench"},
        {"Key": "Component", "Value": "endpoint-autoscaling"},
        {"Key": "Endpoint", "Value": endpoint_name},
    ]


def _reput_alarm_with_description(cw, alarm: dict, description: str) -> None:
    """Re-put the alarm preserving all existing config, swapping in the description.

    put_metric_alarm is a full replace — any field not passed is cleared. We copy
    every field that can round-trip through the API. Read-only fields
    (AlarmArn, StateValue, timestamps, etc.) are dropped.
    """
    passthrough_keys = (
        "AlarmName",
        "ActionsEnabled",
        "OKActions",
        "AlarmActions",
        "InsufficientDataActions",
        "MetricName",
        "Namespace",
        "Statistic",
        "ExtendedStatistic",
        "Dimensions",
        "Period",
        "Unit",
        "EvaluationPeriods",
        "DatapointsToAlarm",
        "Threshold",
        "ComparisonOperator",
        "TreatMissingData",
        "EvaluateLowSampleCountPercentile",
        "Metrics",
        "ThresholdMetricId",
    )
    kwargs = {k: alarm[k] for k in passthrough_keys if k in alarm}
    kwargs["AlarmDescription"] = description
    cw.put_metric_alarm(**kwargs)


def backfill(boto3_session, endpoint_filter: str | None, apply: bool) -> None:
    cw = boto3_session.client("cloudwatch")

    matches: list[tuple[dict, str, str]] = []  # (alarm, endpoint_name, role)
    for page in cw.get_paginator("describe_alarms").paginate(AlarmTypes=["MetricAlarm"]):
        for alarm in page.get("MetricAlarms", []):
            ok, role = _is_workbench_autoscaling_alarm(alarm)
            if not ok:
                continue
            ep = _endpoint_name_from_alarm(alarm)
            if ep is None:
                print(f"  [skip] {alarm['AlarmName']}: no EndpointName dimension")
                continue
            if endpoint_filter and ep != endpoint_filter:
                continue
            matches.append((alarm, ep, role))

    if not matches:
        print("No workbench autoscaling alarms found.")
        return

    print(f"Found {len(matches)} matching alarm(s){' (dry-run)' if not apply else ''}:\n")
    for alarm, ep, role in matches:
        name = alarm["AlarmName"]
        current_desc = alarm.get("AlarmDescription", "") or "<none>"
        new_desc = _description_for(ep, role)
        already_set = current_desc == new_desc
        flag = "skip" if already_set else ("apply" if apply else "would-update")
        print(f"  [{flag}] {name}")
        print(f"          endpoint={ep}  role={role}")
        if not already_set:
            print(f"          desc: {current_desc!r}")
            print(f"            ->  {new_desc!r}")

        if not apply:
            continue

        if not already_set:
            _reput_alarm_with_description(cw, alarm, new_desc)
        # Tags are independent — always re-apply (idempotent, cheap).
        cw.tag_resource(ResourceARN=alarm["AlarmArn"], Tags=_tags_for(ep))

    print(f"\n{'Applied' if apply else 'Dry-run complete'}: {len(matches)} alarm(s) processed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    parser.add_argument("--endpoint", help="Limit to a single endpoint name")
    args = parser.parse_args()

    boto3_session = Meta().boto3_session
    try:
        backfill(boto3_session, args.endpoint, args.apply)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
