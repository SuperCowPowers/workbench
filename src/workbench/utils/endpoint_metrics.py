"""EndpointMetrics is a utility class that fetches metrics for a SageMaker endpoint.

One preset per endpoint type:
  * ``realtime``   — persistent instance-backed endpoints
  * ``serverless`` — ServerlessConfig endpoints (pay-per-invoke, cold starts)
  * ``async``      — AsyncInferenceConfig endpoints (queue-backed, scale-to-zero)

Each preset picks a focused, six-metric set that fits a 2-wide × 3-tall subplot
grid in the dashboard. Metrics specific to one endpoint type (e.g., serverless
concurrency utilization, async backlog) live only in that type's preset.
"""

from datetime import datetime, timedelta, timezone
import pandas as pd

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# ---------------------------------------------------------------------------
# Metric presets — each list is exactly 6 entries so the 2-wide grid is full.
# ---------------------------------------------------------------------------
REALTIME_METRICS = {
    "metrics": [
        "Invocations",
        "ModelLatency",
        "OverheadLatency",
        "CPUUtilization",
        "MemoryUtilization",
        "Invocation5XXErrors",
    ],
    "conversions": {
        "Invocations": 1,
        "ModelLatency": 1e-6,  # microseconds → seconds
        "OverheadLatency": 1e-6,
        "CPUUtilization": 1,
        "MemoryUtilization": 1,
        "Invocation5XXErrors": 1,
    },
    "stats": ["Sum", "Maximum", "Maximum", "Average", "Average", "Sum"],
    "expressions": [],
}

SERVERLESS_METRICS = {
    "metrics": [
        "Invocations",
        "ModelLatency",
        "OverheadLatency",
        "ServerlessConcurrentExecutionsUtilization",
        "ModelSetupTime",  # cold-start — real concern for serverless
        "Invocation5XXErrors",
    ],
    "conversions": {
        "Invocations": 1,
        "ModelLatency": 1e-6,
        "OverheadLatency": 1e-6,
        "ServerlessConcurrentExecutionsUtilization": 100,  # fraction → percent
        "ModelSetupTime": 1e-6,
        "Invocation5XXErrors": 1,
    },
    "stats": ["Sum", "Maximum", "Maximum", "Maximum", "Maximum", "Sum"],
    "expressions": [],
}

ASYNC_METRICS = {
    # Async endpoints publish a DIFFERENT metric set than realtime:
    #   * No `Invocations` — use `InvocationsProcessed`
    #   * No `Invocation5XXErrors` — use `InvocationFailures`
    #   * No `OverheadLatency` — replaced conceptually by `TimeInBacklog`
    #   * Backlog metrics are dimensioned by EndpointName ONLY (no VariantName)
    "metrics": [
        "ApproximateBacklogSize",
        "ApproximateBacklogSizePerInstance",
        "InvocationsProcessed",
        "ModelLatency",
        "InvocationFailures",
        # 6th subplot is the derived InstanceCount (see "expressions" below).
    ],
    "conversions": {
        "ApproximateBacklogSize": 1,
        "ApproximateBacklogSizePerInstance": 1,
        "InvocationsProcessed": 1,
        "ModelLatency": 1e-6,
        "InvocationFailures": 1,
        # Math expressions (see "expressions" below) pass through as-is.
        "InstanceCount": 1,
    },
    # ApproximateBacklogSize uses Average (not Maximum) so the derived
    # InstanceCount expression (backlog / per_instance) is self-consistent —
    # both operands use the same aggregation. Maximum would inflate the ratio
    # when backlog spikes but per_instance is averaged over the period.
    "stats": ["Average", "Average", "Sum", "Maximum", "Sum"],
    # Async backlog metrics are published with only the EndpointName dimension.
    # Querying them with VariantName returns no datapoints.
    "endpoint_only": {"ApproximateBacklogSize", "ApproximateBacklogSizePerInstance"},
    # CloudWatch Metric Math expressions, evaluated alongside the raw metrics.
    # `InstanceCount` is derived — SageMaker doesn't publish an instance-count
    # metric natively, but backlog / backlog-per-instance gives us the divisor.
    # Guarded against divide-by-zero when there's no backlog (reads 0, which
    # matches the scale-to-zero idle state).
    "expressions": [
        {
            # CloudWatch requires expression IDs to match ^[a-z][a-zA-Z0-9_]*$
            # (lowercase first char). The Label becomes the DataFrame column name.
            "id": "instance_count",
            "expression": (
                "IF(m_ApproximateBacklogSizePerInstance > 0, "
                "m_ApproximateBacklogSize / m_ApproximateBacklogSizePerInstance, 0)"
            ),
            "label": "InstanceCount",
        },
    ],
}

_PRESETS = {
    "realtime": REALTIME_METRICS,
    "serverless": SERVERLESS_METRICS,
    "async": ASYNC_METRICS,
}


class EndpointMetrics:
    def __init__(self, preset: str = "realtime"):
        """EndpointMetrics Class.

        Args:
            preset: One of ``"realtime"``, ``"serverless"``, or ``"async"`` —
                selects the appropriate CloudWatch metrics for the endpoint type.
        """
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Expected one of {list(_PRESETS)}")

        self.aws_account_clamp = AWSAccountClamp()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.cloudwatch = self.boto3_session.client("cloudwatch")
        self.start_time = None
        self.end_time = None

        config = _PRESETS[preset]
        self.metrics = config["metrics"]
        self.metric_conversions = config["conversions"]
        self.stats = config["stats"]
        self.expressions = config.get("expressions", [])
        # Metrics in this set are queried with only the EndpointName dimension
        # (no VariantName). Async backlog metrics require this.
        self.endpoint_only = config.get("endpoint_only", set())

    def get_metrics(self, endpoint: str, variant: str = "AllTraffic", days_back: int = 3) -> pd.DataFrame:
        """Get the metric data for a given endpoint.
        Args:
            endpoint (str): The name of the endpoint.
            variant (str): The variant name (default: AllTraffic).
            days_back (int): The number of days back to fetch metrics.
        Returns:
            pd.DataFrame: The metric data in a dataframe.
        """
        # Fetch the metrics
        response = self._fetch_metrics(endpoint=endpoint, variant=variant, days_back=days_back)

        # Parse the response
        metric_data = {}
        for metric in response["MetricDataResults"]:
            metric_name = metric["Label"]

            # Pull timestamps and values
            timestamps = metric["Timestamps"]
            values = metric["Values"]
            values = [round(v * self.metric_conversions[metric_name], 2) for v in values]

            # Add the start and end times to the metric data so that all graphs have the same date range (x-axis)
            timestamps.insert(0, self.end_time)
            timestamps.append(self.start_time - timedelta(hours=1))  # Ensure graph starts at 0
            values.insert(0, 0)
            values.append(0)

            # Create a dataframe and set the index to the timestamps
            metric_df = pd.DataFrame({"timestamps": timestamps, "values": values})
            metric_df.set_index("timestamps", inplace=True, drop=True)

            # Ensure the index is a datetime index
            metric_df.index = pd.to_datetime(metric_df.index, errors="raise", utc=True)

            # Set the metric data dataframe
            metric_data[metric_name] = metric_df

        # Merge the dataframes
        metric_df = self._merge_dataframes(metric_data=metric_data)
        return metric_df

    def _fetch_metrics(self, endpoint: str, variant: str, days_back: int):
        """Internal Method: Fetch metrics from CloudWatch."""
        start_time_str, end_time_str = self._get_time_range(days_back=days_back)
        metric_data_queries = self._get_metric_data_queries(endpoint=endpoint, variant=variant)

        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_queries, StartTime=start_time_str, EndTime=end_time_str
        )
        return response

    def _get_time_range(self, days_back=3):
        """Internal Method: Get the time range for the metrics."""
        now_utc = datetime.now(timezone.utc)
        self.end_time = now_utc
        self.start_time = self.end_time - timedelta(days=days_back)

        # Convert times to strings that the CloudWatch API expects
        end_time_str = self.end_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        start_time_str = self.start_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

        return start_time_str, end_time_str

    def _get_metric_data_queries(self, endpoint: str, variant: str) -> list[dict]:
        """Internal: Get the metric data queries for a given endpoint.
        Args:
            endpoint (str): The name of the endpoint.
            variant (str): The variant name.
        Returns:
            list[dict]: The metric data queries.
        """
        # Period set to 1 hour
        period = 3600
        metric_data_queries = []

        for metric_name, stat in zip(self.metrics, self.stats):
            dims = [{"Name": "EndpointName", "Value": endpoint}]
            if metric_name not in self.endpoint_only:
                dims.append({"Name": "VariantName", "Value": variant})
            query = {
                "Id": f"m_{metric_name}",
                # Explicit Label — CloudWatch otherwise prefixes with the variant
                # name (e.g., "AllTraffic InvocationsProcessed") which breaks the
                # metric_conversions lookup downstream.
                "Label": metric_name,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/SageMaker",
                        "MetricName": metric_name,
                        "Dimensions": dims,
                    },
                    "Period": period,
                    "Stat": stat,
                },
                "ReturnData": True,
            }
            metric_data_queries.append(query)

        # Metric math expressions — evaluated by CloudWatch server-side from the
        # m_* queries above. Label becomes the column name in the returned frame.
        for expr in self.expressions:
            metric_data_queries.append(
                {
                    "Id": expr["id"],
                    "Expression": expr["expression"],
                    "Label": expr["label"],
                    "ReturnData": True,
                }
            )

        return metric_data_queries

    @staticmethod
    def _merge_dataframes(metric_data: dict) -> pd.DataFrame:
        """Internal Method: Merge the metric dataframes into a single dataframe.
        Args:
            metric_data (dict): The metric data as a dictionary of dataframes.
        Returns:
            pd.DataFrame: The merged metric data.
        """
        merged_df = pd.DataFrame()
        for metric_name, df in metric_data.items():
            if merged_df.empty:
                merged_df = df.rename(columns={"values": metric_name})
            else:
                merged_df = pd.merge(
                    merged_df,
                    df.rename(columns={"values": metric_name}),
                    left_index=True,
                    right_index=True,
                    how="outer",
                )

        # Sort by index (timestamp)
        merged_df.sort_index(inplace=True)

        # Resample the index to 1-hour intervals
        merged_df = merged_df.resample("1h").max()

        # Fill NA values with 0 and reset index
        merged_df.fillna(0, inplace=True)
        merged_df.reset_index(inplace=True)
        return merged_df


if __name__ == "__main__":
    """Exercise the EndpointMetrics class."""
    from pprint import pprint

    endpoint = "abalone-regression"
    print(f"Fetching metrics for endpoint: {endpoint}...")

    # Create the Class and query for metrics
    my_metrics = EndpointMetrics()
    metrics_data = my_metrics.get_metrics(endpoint=endpoint, days_back=3)
    pprint(metrics_data)

    # Sum up the columns and display
    print(metrics_data.select_dtypes(include=["number"]).sum())
