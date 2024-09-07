import time
import argparse
from datetime import datetime, timedelta, timezone
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp

# Define the log levels to search for
search_map = {
    "ALL": None,
    "IMPORTANT": ["IMPORTANT", "WARNING", "MONITOR", "ERROR", "CRITICAL"],
    "WARNING": ["WARNING", "MONITOR", "ERROR", "CRITICAL"],
    "MONITOR": ["MONITOR", "ERROR", "CRITICAL"],
    "ERROR": ["ERROR", "CRITICAL"],
}


def get_cloudwatch_client():
    """Get the CloudWatch Logs client using the SageWorks assumed role session."""
    session = AWSAccountClamp().boto3_session
    return session.client("logs")


def get_latest_log_events(client, log_group_name, start_time, end_time=None, stream_filter=None):
    """Retrieve the latest log events from all or filtered log streams in a CloudWatch Logs group."""
    try:
        log_events = []
        next_token = None

        while True:
            # Create the parameters dictionary for describe_log_streams
            stream_params = {
                "logGroupName": log_group_name,
            }

            # Only add nextToken if it's not None
            if next_token:
                stream_params["nextToken"] = next_token

            streams_response = client.describe_log_streams(**stream_params)

            for log_stream in streams_response.get("logStreams", []):
                log_stream_name = log_stream["logStreamName"]

                # Filter streams based on the stream_filter substring
                if stream_filter and stream_filter not in log_stream_name:
                    continue

                # Create the parameters dictionary for get_log_events
                params = {
                    "logGroupName": log_group_name,
                    "logStreamName": log_stream_name,
                    "startTime": int(start_time.timestamp() * 1000),  # Convert to milliseconds
                    "startFromHead": True,  # Start from the earliest log event in the stream
                }

                # Only add endTime if it's not None
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000)

                events_response = client.get_log_events(**params)

                events = events_response.get("events", [])
                for event in events:
                    event["logStreamName"] = log_stream_name

                log_events.extend(events)

            next_token = streams_response.get("nextToken")
            if not next_token:
                break

        return log_events

    except client.exceptions.ResourceNotFoundException:
        print(f"Log group {log_group_name} not found.")
        return []


def merge_ranges(ranges):
    """Merge overlapping or adjacent line ranges."""
    if not ranges:
        return []

    # Sort ranges by start line
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            # If the current range overlaps or is adjacent to the last range, merge them
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def monitor_log_group(
    log_group_name,
    start_time,
    end_time=None,
    poll_interval=10,
    sort_by_stream=False,
    utc_time=False,
    search=None,
    before=10,
    after=0,
    stream_filter=None,
):
    """Continuously monitor the CloudWatch Logs group for new log messages from all log streams."""
    client = get_cloudwatch_client()

    print(f"Monitoring log group: {log_group_name} from {start_time} UTC")

    while True:
        # Get the latest log events with stream filtering if provided
        log_events = get_latest_log_events(client, log_group_name, start_time, end_time, stream_filter)

        if log_events:
            # Sort the events by timestamp by default
            if sort_by_stream:
                log_events.sort(key=lambda x: (x["logStreamName"], x["timestamp"]))
            else:
                log_events.sort(key=lambda x: x["timestamp"])

            # Handle special search terms
            search_terms = search_map.get(search.upper(), [search]) if search else None

            # If search is provided, filter log events and include context
            if search_terms:
                ranges = []
                for i, event in enumerate(log_events):
                    if any(term in event["message"] for term in search_terms):
                        # Calculate the start and end index for this match
                        start_index = max(i - before, 0)
                        end_index = min(i + after, len(log_events) - 1)
                        ranges.append((start_index, end_index))

                # Merge overlapping ranges
                merged_ranges = merge_ranges(ranges)

                # Collect filtered events based on merged ranges
                filtered_events = []
                for start, end in merged_ranges:
                    filtered_events.extend(log_events[start : end + 1])
                    filtered_events.append({"logStreamName": None, "timestamp": None, "message": ""})
                    filtered_events.append({"logStreamName": None, "timestamp": None, "message": ""})

                log_events = filtered_events

            for event in log_events:
                if event["logStreamName"] is None and event["timestamp"] is None:
                    print("")  # Print a blank line
                else:
                    log_stream_name = event["logStreamName"]
                    timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, tz=timezone.utc)

                    # Convert the timestamp to local time if utc_time is False
                    if not utc_time:
                        timestamp = timestamp.astimezone()

                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    message = event["message"].strip()
                    print(f"[{log_stream_name}] [{formatted_time}] {message}")

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
    parser.add_argument(
        "--start-time", type=int, default=60, help="Start time in minutes ago. Default is 60 minutes ago."
    )
    parser.add_argument("--end-time", type=int, help="End time in minutes ago for fetching a range of logs.")
    parser.add_argument(
        "--poll-interval", type=int, default=10, help="Polling interval in seconds. Default is 10 seconds."
    )
    parser.add_argument(
        "--sort-by-stream", action="store_true", help="Sort the log events by stream name instead of timestamp."
    )
    parser.add_argument("--utc-time", action="store_true", help="Display timestamps in UTC instead of local.")
    parser.add_argument("--search", default="ERROR", help="Search term to filter log messages.")
    parser.add_argument("--before", type=int, default=10, help="Number of lines to include before the search match.")
    parser.add_argument("--after", type=int, default=0, help="Number of lines to include after the search match.")
    parser.add_argument("--stream", help="Filter log streams by a substring.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine the start time and end time for log monitoring (in UTC)
    start_time = datetime.now(timezone.utc) - timedelta(minutes=args.start_time)
    end_time = datetime.now(timezone.utc) - timedelta(minutes=args.end_time) if args.end_time else None

    monitor_log_group(
        "SageWorksLogGroup",
        start_time,
        end_time,
        args.poll_interval,
        args.sort_by_stream,
        args.utc_time,
        args.search,
        args.before,
        args.after,
        args.stream,
    )


if __name__ == "__main__":
    main()
