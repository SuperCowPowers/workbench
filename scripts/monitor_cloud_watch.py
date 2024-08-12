import time
import argparse
from datetime import datetime, timedelta, timezone
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


def get_cloudwatch_client():
    """Get the CloudWatch Logs client using the SageWorks assumed role session."""
    session = AWSAccountClamp().boto_session()
    return session.client('logs')


def get_latest_log_events(client, log_group_name, start_time, end_time=None):
    """Retrieve the latest log events from all log streams in a CloudWatch Logs group."""
    try:
        streams_response = client.describe_log_streams(
            logGroupName=log_group_name
        )

        log_events = []

        for log_stream in streams_response.get('logStreams', []):
            log_stream_name = log_stream['logStreamName']
            events_response = client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                startTime=int(start_time.timestamp() * 1000),  # Convert to milliseconds
                startFromHead=False
            )

            events = events_response.get('events', [])
            for event in events:
                event['logStreamName'] = log_stream_name

            log_events.extend(events)

        return log_events

    except client.exceptions.ResourceNotFoundException:
        print(f"Log group {log_group_name} not found.")
        return []


def monitor_log_group(log_group_name, start_time, end_time=None, poll_interval=10):
    """Continuously monitor the CloudWatch Logs group for new log messages from all log streams."""
    client = get_cloudwatch_client()

    print(f"Monitoring log group: {log_group_name} from {start_time}")

    while True:
        # Get the latest log events
        log_events = get_latest_log_events(client, log_group_name, start_time, end_time)

        if log_events:
            for event in log_events:
                log_stream_name = event['logStreamName']
                timestamp = datetime.utcfromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                message = event['message'].strip()
                print(f"[{log_stream_name}] [{timestamp}] {message}")

            # Update the start time to just after the last event's timestamp
            if end_time is None:
                start_time = datetime.now(timezone.utc)
            else:
                # Exit the loop after fetching logs for the specified range
                break

        # Wait for the next poll if monitoring real-time logs
        if end_time is None:
            time.sleep(poll_interval)
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor CloudWatch Logs.")
    parser.add_argument("--log-group", default="SageWorksLogGroup", help="The CloudWatch Logs group name (default: SageWorksLogGroup).")
    parser.add_argument("--start-time", type=int, help="Start time in minutes ago. Default is 5 minutes ago.")
    parser.add_argument("--end-time", type=int, help="End time in minutes ago for fetching a range of logs.")
    parser.add_argument("--poll-interval", type=int, default=10, help="Polling interval in seconds. Default is 10 seconds.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine the start time and end time for log monitoring
    start_time = datetime.now(timezone.utc) - timedelta(minutes=args.start_time or 5)
    end_time = datetime.now(timezone.utc) - timedelta(minutes=args.end_time) if args.end_time else None

    monitor_log_group(args.log_group, start_time, end_time, args.poll_interval)
