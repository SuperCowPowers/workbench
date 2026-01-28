"""
Lambda handler for Batch job failure notifications.

Triggered by EventBridge when a Batch job fails. Fetches CloudWatch logs
and sends a detailed SNS notification including environment info and error details.
"""

import os
import json
import boto3

sns = boto3.client("sns")
logs = boto3.client("logs")


def get_log_events(log_stream_name: str, limit: int = 50) -> list[str]:
    """
    Fetch recent log events from CloudWatch Logs.

    Args:
        log_stream_name: The log stream name (from Batch job details)
        limit: Maximum number of log events to fetch

    Returns:
        List of log message strings
    """
    # Batch jobs log to /aws/batch/job
    log_group = "/aws/batch/job"

    try:
        response = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream_name,
            limit=limit,
            startFromHead=False,  # Get most recent events
        )
        events = response.get("events", [])
        return [event["message"] for event in events]
    except logs.exceptions.ResourceNotFoundException:
        return ["(Log stream not found)"]
    except Exception as e:
        return [f"(Error fetching logs: {e})"]


def format_log_messages(messages: list[str], max_chars: int = 10000) -> str:
    """
    Format log messages for SNS, truncating if necessary.

    Args:
        messages: List of log message strings
        max_chars: Maximum total characters to include

    Returns:
        Formatted log output string
    """
    if not messages:
        return "(No log messages available)"

    output = "\n".join(messages)

    if len(output) > max_chars:
        output = output[-max_chars:]
        output = "... (truncated)\n" + output

    return output


def lambda_handler(event, context):
    """
    Handle Batch job failure event from EventBridge.

    Extracts job details, fetches CloudWatch logs, and sends SNS notification.
    """
    print(f"Received event: {json.dumps(event, indent=2)}")

    # Extract job details from the event
    detail = event.get("detail", {})

    job_name = detail.get("jobName", "Unknown")
    job_id = detail.get("jobId", "Unknown")
    status = detail.get("status", "Unknown")
    status_reason = detail.get("statusReason", "Unknown")

    # Get container details for log stream
    container = detail.get("container", {})
    log_stream_name = container.get("logStreamName", "")

    # Get environment variables from the job
    env_vars = {env["name"]: env["value"] for env in container.get("environment", [])}
    script_path = env_vars.get("ML_PIPELINE_S3_PATH", "N/A")

    # Environment info
    environment = os.environ.get("WORKBENCH_ENVIRONMENT", "unknown")
    region = event.get("region") or os.environ.get("AWS_REGION")
    topic_arn = os.environ.get("BATCH_FAILURE_TOPIC_ARN")

    # Fetch CloudWatch logs
    log_messages = []
    if log_stream_name:
        log_messages = get_log_events(log_stream_name, limit=200)

    # Build the console URL
    console_url = f"https://{region}.console.aws.amazon.com/batch/home?region={region}" f"#jobs/fargate/detail/{job_id}"

    # Build the notification
    subject = f"Batch Job Failed: {environment}: {job_name}"

    # Truncate subject if too long (SNS limit is 100 chars)
    if len(subject) > 100:
        subject = subject[:97] + "..."

    message_body = f"""Batch Job Failed!

Environment: {environment}
Job Name: {job_name}
Job ID: {job_id}
Status: {status}
Reason: {status_reason}
Script: {script_path}

View job details:
{console_url}

=== Recent Log Output ===

{format_log_messages(log_messages)}
"""

    # Publish to SNS
    if topic_arn:
        try:
            response = sns.publish(
                TopicArn=topic_arn,
                Subject=subject,
                Message=message_body,
            )
            print(f"Published notification to SNS (MessageId: {response['MessageId']})")
        except Exception as e:
            print(f"Error publishing to SNS: {e}")
    else:
        print("Warning: BATCH_FAILURE_TOPIC_ARN not set, skipping SNS notification")
        print(f"Would have sent:\nSubject: {subject}\n\n{message_body}")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Batch failure notification sent",
                "job_name": job_name,
                "job_id": job_id,
                "environment": environment,
            }
        ),
    }
