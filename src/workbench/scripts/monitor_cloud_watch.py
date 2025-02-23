import signal
import atexit
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.repl_utils import cprint, Spinner

# Define the log levels to include all log levels above the specified level
log_level_map = {
    "all": [],
    "important": ["important", "warning", "monitor", "error", "critical"],
    "warning": ["warning", "monitor", "error", "critical"],
    "monitor": ["monitor", "error", "critical"],
    "error": ["error", "critical"],
}

# Global flag to display timestamps in local time
local_time = True


def date_display(dt):
    """Convert datetime to a concise human-readable format

    Args:
        dt (datetime): The datetime object to format.
    """
    if local_time:
        dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %I:%M%p")
    else:
        return dt.strftime("%Y-%m-%d %I:%M%p") + "(UTC)"


def get_cloudwatch_client():
    """Get the CloudWatch Logs client using the Workbench assumed role session."""
    session = AWSAccountClamp().boto3_session
    return session.client("logs")


def get_active_log_streams(client, log_group_name, start_time_ms, stream_filter=None):
    """Retrieve log streams that have events after the specified start time."""

    # Get all the streams in the log group
    active_streams = []
    stream_params = {
        "logGroupName": log_group_name,
        "orderBy": "LastEventTime",
        "descending": True,
    }

    # Loop to retrieve all log streams (maximum 50 per call)
    while True:
        response = client.describe_log_streams(**stream_params)
        log_streams = response.get("logStreams", [])

        for log_stream in log_streams:
            log_stream_name = log_stream["logStreamName"]
            last_event_timestamp = log_stream.get("lastEventTimestamp")

            # Include streams with events since the specified start time
            # Note: There's some issue where the last event timestamp is 'off'
            #       so we're going to add 60 minutes from the last event timestamp
            last_event_timestamp += 60 * 60 * 1000
            if last_event_timestamp >= start_time_ms:
                active_streams.append(log_stream_name)
            else:
                break  # Stop if we reach streams older than the start time

        # Check if there are more streams to retrieve
        if "nextToken" in response:
            stream_params["nextToken"] = response["nextToken"]
        else:
            break

    # Sort and report the active log streams
    active_streams.sort()
    if active_streams:
        print("Active log streams:", len(active_streams))

    # Filter the active streams by a substring if provided
    if stream_filter and active_streams:
        print(f"Filtering active log streams by '{stream_filter}'...")
        active_streams = [stream for stream in active_streams if stream_filter in stream]

    for stream in active_streams:
        print(f"\t - {stream}")

    # Return the active log streams
    return active_streams


def get_latest_log_events(client, log_group_name, start_time, end_time=None, stream_filter=None):
    """Retrieve the latest log events from the active/filtered log streams in a CloudWatch Logs group."""

    # Initialize first run attribute
    if not hasattr(get_latest_log_events, "first_run"):
        get_latest_log_events.first_run = True

    log_events = []
    start_time_ms = int(start_time.timestamp() * 1000)  # Convert start_time to milliseconds

    # Get the active log streams with events since start_time
    active_streams = get_active_log_streams(client, log_group_name, start_time_ms, stream_filter)
    if active_streams:
        print(f"Processing log events from {date_display(start_time)} on {len(active_streams)} active log streams...")
        get_latest_log_events.first_run = False
    else:
        if get_latest_log_events.first_run:
            print(f"No active log streams (start_time:{date_display(start_time)}, stream-filter:'{stream_filter}')")
            get_latest_log_events.first_run = False
        else:
            print("Monitoring for new events...")
        return log_events

    # Iterate over the active streams and fetch log events
    for log_stream_name in active_streams:
        params = {
            "logGroupName": log_group_name,
            "logStreamName": log_stream_name,
            "startTime": start_time_ms,  # Use start_time in milliseconds
            "startFromHead": True,  # Start from the nearest event to start_time
        }
        next_event_token = None
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        # Process the log events from this log stream
        spinner = Spinner("lightpurple", f"Pulling events from {log_stream_name}:")
        spinner.start()
        log_stream_events = 0

        # Get the log events for the active log stream
        while True:
            if next_event_token:
                params["nextToken"] = next_event_token
                params.pop("startTime", None)  # Remove startTime when using nextToken

            # Fetch the log events (this call takes a while: optimize if we can)
            events_response = client.get_log_events(**params)

            events = events_response.get("events", [])
            for event in events:
                event["logStreamName"] = log_stream_name

            # Add the log stream events to our list of all log events
            log_stream_events += len(events)
            log_events.extend(events)

            # Handle pagination for log events
            next_event_token = events_response.get("nextForwardToken")

            # Break the loop if there are no more events to fetch
            if not next_event_token or next_event_token == params.get("nextToken"):
                spinner.stop()
                print(f"Processed {log_stream_events} events from {log_stream_name} (Total: {len(log_events)})")
                break

    # Return the log events
    return log_events


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
    log_level=None,
    search_terms=None,
    before=10,
    after=0,
    stream_filter=None,
):
    """Continuously monitor the CloudWatch Logs group for new log messages from all log streams."""
    client = get_cloudwatch_client()

    # Convert any search terms to lowercase
    search_terms = [term.lower() for term in search_terms] if search_terms else []

    # Handle log levels (typically if a search term is used we should use 'all' log levels
    if search_terms:
        log_level = "all"
    log_levels = log_level_map.get(log_level.lower(), [log_level]) if log_level else []

    print(f"Monitoring log group: {log_group_name} from {date_display(start_time)}")
    print(f"Log levels: {log_levels}")
    print(f"Search terms: {search_terms}")
    while True:
        # Get the latest log events with stream filtering if provided
        all_log_events = get_latest_log_events(client, log_group_name, start_time, end_time, stream_filter)

        # Match log levels and search terms, collect ranges, and merge overlapping ranges
        ranges = []
        for i, event in enumerate(all_log_events):
            # Match any of log levels in the log message
            if not log_levels or any(term in event["message"].lower() for term in log_levels):

                # Check the search terms
                if not search_terms or any(term in event["message"].lower() for term in search_terms):

                    # Calculate the start and end index for this match
                    start_index = max(i - before, 0)
                    end_index = min(i + after, len(all_log_events) - 1)
                    ranges.append((start_index, end_index))

        # Merge overlapping ranges
        merged_ranges = merge_ranges(ranges)

        # Collect filtered events based on merged ranges
        filtered_events = []
        for start, end in merged_ranges:
            filtered_events.extend(all_log_events[start : end + 1])

            # These are just blank lines to separate the log message 'groups'
            filtered_events.append({"logStreamName": None, "timestamp": None, "message": ""})
            filtered_events.append({"logStreamName": None, "timestamp": None, "message": ""})

        # Display the filtered log events
        if not filtered_events:
            print("No log events found, matching the specified criteria...")
        for event in filtered_events:
            if event["logStreamName"] is None and event["timestamp"] is None:
                print("")
            else:
                log_stream_name = event["logStreamName"]
                timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, tz=timezone.utc)
                message = event["message"].strip()
                print(f"[{log_stream_name}] [{date_display(timestamp)}] {message}")

        # Update the start time to just after the last event's timestamp
        if end_time is None:
            start_time = datetime.now(timezone.utc)
        else:
            break

        # Wait for the next poll if monitoring realtime logs
        if end_time is None:
            time.sleep(poll_interval)
        else:
            break


# Function to handle SIGINT (Control-C)
def signal_handler(sig, frame):
    sys.exit(0)


# Function to handle cleanup tasks at exit
def cleanup():
    cprint("lightgreen", "\nGoodbye from Workbench CloudWatch!")


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor CloudWatch Logs.")
    parser.add_argument(
        "--start-time",
        type=int,
        default=600,
        help="Start time in minutes ago. Default is 600 minutes ago.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        help="End time in minutes ago for fetching a range of logs.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds. Default is 60 seconds.",
    )
    parser.add_argument(
        "--utc-time",
        action="store_true",
        help="Display timestamps in UTC instead of local.",
    )
    parser.add_argument("--log-level", default="error", help="Log level to filter log messages.")
    parser.add_argument("--search", nargs="+", help="Search terms to filter log messages.")
    parser.add_argument(
        "--before",
        type=int,
        default=10,
        help="Number of lines to include before the log level match.",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=0,
        help="Number of lines to include after the log level match.",
    )
    parser.add_argument("--stream", help="Filter log streams by a substring.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Register the signal handler for SIGINT (Control-C)
    signal.signal(signal.SIGINT, signal_handler)

    # Register the cleanup function to be called at exit
    atexit.register(cleanup)

    # Determine the start time and end time for log monitoring (in UTC)
    start_time = datetime.now(timezone.utc) - timedelta(minutes=args.start_time)
    end_time = datetime.now(timezone.utc) - timedelta(minutes=args.end_time) if args.end_time else None

    # Set the global flag to display timestamps in local time
    global local_time
    local_time = not args.utc_time

    # Monitor the CloudWatch Logs group
    monitor_log_group(
        "WorkbenchLogGroup",
        start_time=start_time,
        end_time=end_time,
        poll_interval=args.poll_interval,
        log_level=args.log_level,
        search_terms=args.search,
        before=args.before,
        after=args.after,
        stream_filter=args.stream,
    )


if __name__ == "__main__":
    main()
