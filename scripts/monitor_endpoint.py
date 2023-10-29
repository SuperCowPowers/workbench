# Description: This script is used to monitor the number of invocations of a SageMaker endpoint over the last N days.
from datetime import datetime, timedelta
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

# Grab our boto3 session from the Account Clamp
boto_session = AWSAccountClamp().boto3_session
cloudwatch = boto_session.client('cloudwatch')

endpoint = 'abalone-regression-end'

# Get midnight tomorrow in UTC
now_utc = datetime.utcnow()
today_time = datetime(now_utc.year, now_utc.month, now_utc.day)
end_time = today_time + timedelta(days=1)

# Define the number of days to go back
N = 7

# Calculate start time by subtracting N days from end time
start_time = end_time - timedelta(days=N)

# Convert to the desired string format
end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
print(f"Endpoint: {endpoint}")
print(f"Start Time: {start_time_str}")
print(f"End Time: {end_time_str}")


response = cloudwatch.get_metric_data(
    MetricDataQueries=[
        {
            'Id': 'm1',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'Invocations',
                    'Dimensions': [
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint,
                        },
                        {
                            'Name': 'VariantName',
                            'Value': 'AllTraffic',
                        },
                    ]
                },
                'Period': 3600,
                'Stat': 'Sum',
            },
            'ReturnData': True,
        },
    ],
    StartTime=start_time_str,
    EndTime=end_time_str
)
print(response)
