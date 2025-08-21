# cloudwatch_utils.py
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Generator
from urllib.parse import quote
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


def get_cloudwatch_client():
    """Get the CloudWatch Logs client using the Workbench assumed role session."""
    session = AWSAccountClamp().boto3_session
    return session.client("logs")


def get_cloudwatch_logs_url(log_group: Optional[str] = None, log_stream: Optional[str] = None) -> Optional[str]:
    """
    Generate CloudWatch logs URL for the current ECS task or specified log group/stream.

    Args:
        log_group: Optional log group name. If not provided, uses AWS_LOGS_GROUP env var or default.
        log_stream: Optional log stream name. If not provided, uses AWS_LOGS_STREAM env var.

    Returns:
        CloudWatch console URL or None if unable to generate
    """
    try:
        # Get AWS region and account from AWSAccountClamp
        aws_clamp = AWSAccountClamp()
        region = aws_clamp.region

        # Get log group and stream
        if not log_group:
            log_group = os.environ.get("AWS_LOGS_GROUP", "/aws/batch/job")
        if not log_stream:
            log_stream = os.environ.get("AWS_LOGS_STREAM")

        if log_stream:
            # URL encode the log stream name
            encoded_stream = quote(log_stream, safe="")
            encoded_group = quote(log_group, safe="")
            url = f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{encoded_group}/log-events/{encoded_stream}"
        else:
            # Fallback to log group view
            encoded_group = quote(log_group, safe="")
            url = f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{encoded_group}"

        return url
    except Exception:  # noqa: BLE001
        # Silently fail - don't want to break the main pipeline for URL generation
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
