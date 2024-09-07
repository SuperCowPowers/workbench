"""DashboardMetrics is a utility class that fetches metrics for the SageWorks Dashboard."""

from datetime import datetime, timedelta, timezone
import pandas as pd

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.ecs_info import ECSInfo


class DashboardMetrics:
    # Get the ECS Cluster Name, Service Name, and Load Balancer Name
    ecs_info = ECSInfo()
    cluster_name = ecs_info.get_cluster_name()
    service_name = ecs_info.get_service_name()
    load_balancer_name = ecs_info.get_load_balancer_name()

    def __init__(self):
        """
        DashboardMetrics Class
        - CPUUtilization: The percentage of CPU utilization.
        - MemoryUtilization: The percentage of memory utilization.
        - ActiveConnectionCount
        - TargetResponseTime
        - RequestCount
        - ProcessedBytes
        - HTTPCode_Target_4XX_Count: The number of 4XX errors returned by the targets.
        - HTTPCode_Target_5XX_Count: The number of 5XX errors returned by the targets.
        """
        self.aws_account_clamp = AWSAccountClamp()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.cloudwatch = self.boto3_session.client("cloudwatch")
        self.start_time = None
        self.end_time = None

    def get_time_range(self, days_back=7):
        now_utc = datetime.now(timezone.utc)
        self.end_time = now_utc
        self.start_time = self.end_time - timedelta(days=days_back)

        # Convert times to strings that the CloudWatch API expects
        end_time_str = self.end_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        start_time_str = self.start_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

        return start_time_str, end_time_str

    def get_ecs_metric_queries(self) -> list[dict]:
        period = 3600  # Hardcoded to 1 hour for now
        metric_data_queries = []

        for metric_name in ["CPUUtilization", "MemoryUtilization"]:
            query = {
                "Id": f"m_{metric_name}",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/ECS",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "ServiceName", "Value": self.service_name},
                            {"Name": "ClusterName", "Value": self.cluster_name},
                        ],
                    },
                    "Period": period,
                    "Stat": "Maximum",
                },
                "ReturnData": True,
            }
            metric_data_queries.append(query)
        return metric_data_queries

    def get_elb_metric_queries(self) -> list[dict]:
        period = 3600  # Hardcoded to 1 hour for now
        metric_data_queries = []

        metrics = [
            "ActiveConnectionCount",
            "TargetResponseTime",
            "RequestCount",
            "ProcessedBytes",
            "HTTPCode_Target_4XX_Count",
            "HTTPCode_Target_5XX_Count",
        ]
        stats = ["Sum", "Maximum", "Sum", "Sum", "Sum", "Sum"]

        for metric, stat in zip(metrics, stats):
            query = {
                "Id": f"m_{metric}",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/ApplicationELB",
                        "MetricName": metric,
                        "Dimensions": [{"Name": "LoadBalancer", "Value": self.load_balancer_name}],
                    },
                    "Period": period,
                    "Stat": stat,
                },
                "ReturnData": True,
            }
            metric_data_queries.append(query)
        return metric_data_queries

    def get_metrics(self, days_back: int = 3) -> pd.DataFrame:
        """Get the metric data for a given service_name
        Args:
            days_back(int): The number of days back to fetch metrics
        Returns:
            pd.DataFrame: The metric data in a dataframe
        """
        # Fetch the metrics
        response = self._fetch_metrics(days_back=days_back)

        # Parse the response
        metric_data = {}
        for metric in response["MetricDataResults"]:
            metric_name = metric["Label"].split(" ")[-1]

            # Pull timestamps and values
            timestamps = metric["Timestamps"]
            values = metric["Values"]
            values = [round(v, 2) for v in values]

            # We're going to add the start and end times to the metric data so that
            # every graph has the same date range (x-axis)
            timestamps.insert(0, self.end_time)
            timestamps.append(self.start_time - timedelta(hours=1))  # Make sure graph starts at 0
            values.insert(0, 0)
            values.append(0)

            # Create a dataframe and set the index to the timestamps
            metric_df = pd.DataFrame({"timestamps": timestamps, "values": values})
            metric_df.set_index("timestamps", inplace=True, drop=True)

            # Make sure the index is a datetime index
            metric_df.index = pd.to_datetime(metric_df.index, errors="raise", utc=True)

            # Set our metric data dataframe
            metric_data[metric_name] = metric_df

        # Now we're going to merge the dataframes
        metric_df = self._merge_dataframes(metric_data=metric_data)
        return metric_df

    @staticmethod
    def _merge_dataframes(metric_data: dict) -> pd.DataFrame:
        """Internal Method: Merge the metric dataframes into a single dataframe
        Args:
            metric_data(dict): The metric data in as a dictionary of dataframes
        Returns:
            pd.DataFrame: The merged metric data
        """
        # Merge DataFrames from the dictionary
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

        # Sort by index (which is the timestamp)
        merged_df.sort_index(inplace=True)

        # Resample the index to have 1 hour intervals
        merged_df = merged_df.resample("1H").max()

        # Fill NA values with 0 and reset the index (so we can serialize to JSON)
        merged_df.fillna(0, inplace=True)
        merged_df.reset_index(inplace=True)
        return merged_df

    def _fetch_metrics(self, days_back: int):
        """Internal Method: Fetch metrics from CloudWatch"""
        start_time_str, end_time_str = self.get_time_range(days_back=days_back)
        metric_data_queries = self.get_ecs_metric_queries() + self.get_elb_metric_queries()

        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_queries, StartTime=start_time_str, EndTime=end_time_str
        )
        return response


if __name__ == "__main__":
    """Exercise the DashboardMetrics class"""
    from pprint import pprint

    # Create the Class and query for metrics
    my_metrics = DashboardMetrics()
    metrics_data = my_metrics.get_metrics(days_back=3)
    pprint(metrics_data)
