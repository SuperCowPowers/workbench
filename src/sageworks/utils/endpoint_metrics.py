"""EndpointMetrics is a utility class that fetches metrics for a SageMaker endpoint."""
from datetime import datetime, timedelta

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class EndpointMetrics:
    def __init__(self):
        """
        EndpointMetrics Class
        - Invocations: Number of times the endpoint was invoked.
        - ModelLatency:Time taken by just the model to make a prediction.
        - OverheadLatency: Total time from request until response (does NOT include ModelLatency)
        - ModelSetupTime: Time to launch new compute resources for a serverless endpoint.
        - InvocationModelErrors: The total number of errors (non 200 response codes)
        - Invocation5XXErrors: The number of server error status codes returned.
        - Invocation4XXErrors: The number of client error status codes returned.
        """
        self.aws_account_clamp = AWSAccountClamp()
        self.boto_session = self.aws_account_clamp.boto_session()
        self.cloudwatch = self.boto_session.client('cloudwatch')
        self.metrics = ['Invocations', 'ModelLatency', 'OverheadLatency', 'ModelSetupTime',
                        'InvocationModelErrors', 'Invocation5XXErrors', 'Invocation4XXErrors']
        self.metric_conversions = {"Invocations": 1, "ModelLatency": 1e-6, "OverheadLatency": 1e-6,
                                   "ModelSetupTime": 1e-6, "InvocationModelErrors": 1,
                                   "Invocation5XXErrors": 1, "Invocation4XXErrors": 1}
        self.stats = ['Sum', 'Average', 'Average', 'Average', 'Sum', 'Sum', 'Sum']

    def get_time_range(self, days_back=7):
        now_utc = datetime.utcnow()
        today_time = datetime(now_utc.year, now_utc.month, now_utc.day)
        end_time = today_time + timedelta(days=1)
        start_time = end_time - timedelta(days=days_back)

        end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

        return start_time_str, end_time_str

    def get_metric_data_queries(self, endpoint, days_back=1):

        # Change the Period based on the number of days back
        if days_back <= 1:
            period = 600
        elif days_back <= 7:
            period = 3600
        metric_data_queries = []

        for metric_name, stat in zip(self.metrics, self.stats):
            query = {
                'Id': f'm_{metric_name}',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': metric_name,
                        'Dimensions': [
                            {'Name': 'EndpointName', 'Value': endpoint},
                            {'Name': 'VariantName', 'Value': 'AllTraffic'}
                        ]
                    },
                    'Period': period,
                    'Stat': stat,
                },
                'ReturnData': True
            }
            metric_data_queries.append(query)

        return metric_data_queries

    def get_metrics(self, endpoint: str, days_back: int = 7) -> dict:
        """Get the metric data for a given endpoint
        Args:
            endpoint(str): The name of the endpoint
            days_back(int): The number of days back to fetch metrics
        Returns:
            dict: The metric data in the following form
                  {'metric_name_1': {'timestamps': , 'values': [metric_values]},
                   'metric_name_2': {'timestamps': , 'values': [metric_values]},
                     ...}
        """
        # Fetch the metrics
        response = self._fetch_metrics(endpoint=endpoint, days_back=days_back)

        # Parse the response
        metric_data = {}
        for metric in response['MetricDataResults']:
            metric_name = metric['Label']
            timestamps = metric['Timestamps']
            values = metric['Values']
            values = [round(v * self.metric_conversions[metric_name], 2) for v in values]
            metric_data[metric_name] = {'timestamps': timestamps, 'values': values}

        return metric_data

    def _fetch_metrics(self, endpoint: str, days_back: int):
        """Internal Method: Fetch metrics from CloudWatch"""
        start_time_str, end_time_str = self.get_time_range(days_back=days_back)
        metric_data_queries = self.get_metric_data_queries(endpoint=endpoint, days_back=days_back)

        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_queries,
            StartTime=start_time_str,
            EndTime=end_time_str
        )
        return response


if __name__ == "__main__":
    """Exercise the EndpointMetrics class"""
    from pprint import pprint

    # Create the Class and query for metrics
    my_metrics = EndpointMetrics()
    metrics_data = my_metrics.get_metrics(endpoint='abalone-regression-end')
    pprint(metrics_data)
