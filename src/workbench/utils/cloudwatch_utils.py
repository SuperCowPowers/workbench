"""AWS CloudWatch utility functions for Workbench."""

import time
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Generator
from urllib.parse import quote
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

log = logging.getLogger("workbench")


def get_cloudwatch_client():
    """Get the CloudWatch Logs client using the Workbench assumed role session."""
    session = AWSAccountClamp().boto3_session
    return session.client("logs")


def get_cloudwatch_logs_url(log_group: str, log_stream: str) -> Optional[str]:
    """
    Generate CloudWatch logs URL for the specified log group and stream.

    Args:
        log_group: Log group name (e.g., '/aws/batch/job')
        log_stream: Log stream name

    Returns:
        CloudWatch console URL or None if unable to generate
    """
    try:
        region = AWSAccountClamp().region

        # URL encode the log group and stream
        encoded_group = quote(log_group, safe="")
        encoded_stream = quote(log_stream, safe="")

        return (
            f"https://{region}.console.aws.amazon.com/cloudwatch/home?"
            f"region={region}#logsV2:log-groups/log-group/{encoded_group}"
            f"/log-events/{encoded_stream}"
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"Failed to generate CloudWatch logs URL: {e}")
        return None


def get_active_log_streams(
    log_group_name: str, start_time_ms: int, stream_filter: Optional[str] = None, client=None
) -> List[str]:
    """Retrieve log streams that have events after the specified start time."""
    if not client:
        client = get_cloudwatch_client()
    active_streams = []
    stream_params = {
        "logGroupName": log_group_name,
        "orderBy": "LastEventTime",
        "descending": True,
    }
    while True:
        response = client.describe_log_streams(**stream_params)
        log_streams = response.get("logStreams", [])
        for log_stream in log_streams:
            log_stream_name = log_stream["logStreamName"]
            last_event_timestamp = log_stream.get("lastEventTimestamp", 0)
            if last_event_timestamp >= start_time_ms:
                active_streams.append(log_stream_name)
            else:
                break
        if "nextToken" in response:
            stream_params["nextToken"] = response["nextToken"]
        else:
            break
    # Sort and filter streams
    active_streams.sort()
    if stream_filter and active_streams:
        active_streams = [stream for stream in active_streams if stream_filter in stream]
    return active_streams


def stream_log_events(
    log_group_name: str,
    log_stream_name: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    follow: bool = False,
    client=None,
) -> Generator[Dict, None, None]:
    """
    Stream log events from a specific log stream.
    Yields:
        Log events as dictionaries
    """
    if not client:
        client = get_cloudwatch_client()
    params = {"logGroupName": log_group_name, "logStreamName": log_stream_name, "startFromHead": True}
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)
    next_token = None
    while True:
        if next_token:
            params["nextToken"] = next_token
            params.pop("startTime", None)
        try:
            response = client.get_log_events(**params)
            events = response.get("events", [])
            for event in events:
                event["logStreamName"] = log_stream_name
                yield event
            next_token = response.get("nextForwardToken")
            # Break if no more events or same token
            if not next_token or next_token == params.get("nextToken"):
                if not follow:
                    break
                time.sleep(2)
        except client.exceptions.ResourceNotFoundException:
            if not follow:
                break
            time.sleep(2)


def print_log_event(
    event: dict, show_stream: bool = True, local_time: bool = True, custom_format: Optional[str] = None
):
    """Print a formatted log event."""
    timestamp = datetime.fromtimestamp(event["timestamp"] / 1000, tz=timezone.utc)
    if local_time:
        timestamp = timestamp.astimezone()
    message = event["message"].rstrip()
    if custom_format:
        # Allow custom formatting
        print(custom_format.format(stream=event.get("logStreamName", ""), time=timestamp, message=message))
    elif show_stream and "logStreamName" in event:
        print(f"[{event['logStreamName']}] [{timestamp:%Y-%m-%d %I:%M%p}] {message}")
    else:
        print(f"[{timestamp:%H:%M:%S}] {message}")
