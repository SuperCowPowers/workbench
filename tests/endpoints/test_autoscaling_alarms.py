"""Alarms behind async endpoint autoscaling."""

from unittest.mock import MagicMock

from workbench.utils.endpoint_autoscaling import _install_batch


def _alarms() -> dict:
    aas, cw = MagicMock(), MagicMock()
    aas.put_scaling_policy.return_value = {"PolicyARN": "arn:aws:autoscaling:policy"}
    _install_batch(aas, cw, "ep", "endpoint/ep/variant/AllTraffic", 0, 8, 60)
    return {call.kwargs["AlarmName"]: call.kwargs for call in cw.put_metric_alarm.call_args_list}


def test_scale_out_fires_on_the_first_minute_of_backlog():
    alarm = _alarms()["ep-scale-out"]
    assert alarm["MetricName"] == "ApproximateBacklogSize"
    assert alarm["EvaluationPeriods"] == 1
    assert alarm["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"


def test_scale_in_counts_only_minutes_an_instance_is_up():
    alarm = _alarms()["ep-scale-in"]
    assert "MetricName" not in alarm  # metric math excludes the single-metric fields
    assert alarm["EvaluationPeriods"] == alarm["DatapointsToAlarm"] == 60
    assert alarm["ComparisonOperator"] == "LessThanThreshold"

    metrics = {metric["Id"]: metric for metric in alarm["Metrics"]}
    assert metrics["idle"]["Expression"] == "IF(FILL(cpu, -1) >= 0, FILL(backlog, 0), 1)"
    assert [m["Id"] for m in alarm["Metrics"] if m["ReturnData"]] == ["idle"]

    cpu = metrics["cpu"]["MetricStat"]["Metric"]
    assert cpu["MetricName"] == "CPUUtilization"
    assert {"Name": "VariantName", "Value": "AllTraffic"} in cpu["Dimensions"]
