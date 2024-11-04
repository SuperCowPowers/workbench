import os
import logging
from datetime import datetime, timezone
import getpass
import time  # For managing send intervals

# SageWorks imports
from sageworks.utils.execution_environment import (
    running_on_lambda,
    running_on_glue,
    running_on_ecs,
    running_on_docker,
    glue_job_name,
    glue_job_run_id,
    ecs_job_name,
)
from sageworks.core.cloud_platform.aws.aws_session import AWSSession


class CloudWatchHandler(logging.Handler):
    """A helper class to add a CloudWatch Logs handler with buffering to a logger"""

    def __init__(self, buffer_size=10, send_interval=5):
        super().__init__()  # Initialize the base Handler class
        from sageworks.utils.sageworks_logging import ColoredFormatter  # Import here to avoid circular imports

        self.boto3_session = AWSSession().boto3_session
        self.log_stream_name = self.determine_log_stream()
        self.formatter = ColoredFormatter("(%(filename)s:%(lineno)d) %(levelname)s %(message)s")
        self.cloudwatch_client = self.boto3_session.client("logs")
        self.sequence_token = None
        self.log_group_name = "SageWorksLogGroup"

        # Buffer to hold log messages
        self.buffer = []
        self.buffer_size = buffer_size
        self.send_interval = send_interval
        self.last_sent_time = time.time()

        # Create the log group and stream
        self.create_log_group()
        self.create_log_stream()

    def emit(self, record):
        """Add a log message to the buffer and send when ready"""
        message = self.format(record)
        log_event = {"timestamp": int(record.created * 1000), "message": message}
        self.buffer.append(log_event)

        # Check if the buffer is full or if the time interval has passed
        if len(self.buffer) >= self.buffer_size or (time.time() - self.last_sent_time) >= self.send_interval:
            self.send_logs()

    def send_logs(self):
        """Send buffered log messages to CloudWatch"""
        if not self.buffer:
            return  # Nothing to send

        # Sort the buffer by timestamp to ensure chronological order
        self.buffer.sort(key=lambda event: event["timestamp"])

        log_event = {
            "logGroupName": self.log_group_name,
            "logStreamName": self.log_stream_name,
            "logEvents": self.buffer,
        }

        if self.sequence_token:
            log_event["sequenceToken"] = self.sequence_token

        try:
            response = self.cloudwatch_client.put_log_events(**log_event)
            self.sequence_token = response.get("nextSequenceToken")
            self.buffer.clear()  # Clear the buffer after successful send
        except self.cloudwatch_client.exceptions.InvalidSequenceTokenException as e:
            # Extract token and retry only the put_log_events call
            self.sequence_token = e.response["Error"]["Message"].split()[-1]
            log_event["sequenceToken"] = self.sequence_token
            self.cloudwatch_client.put_log_events(**log_event)  # Retry without recursion
            self.buffer.clear()
        except Exception as e:
            logging.error(f"Failed to send logs to CloudWatch: {e}")

        # Update last sent time after successful send
        self.last_sent_time = time.time()

    def flush(self):
        """Ensure all logs are sent"""
        self.send_logs()  # Flush remaining logs in the buffer

    def create_log_group(self):
        """Create CloudWatch Log Group if it doesn't exist"""
        try:
            self.cloudwatch_client.create_log_group(logGroupName=self.log_group_name)
        except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
            pass

    def create_log_stream(self):
        """Create CloudWatch Log Stream if it doesn't exist"""
        try:
            self.cloudwatch_client.create_log_stream(
                logGroupName=self.log_group_name, logStreamName=self.log_stream_name
            )
        except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
            pass

    def determine_log_stream(self):
        """Determine the log stream name based on the environment."""
        if running_on_lambda():
            job_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            return f"lambda/{job_name}"
        elif running_on_glue():
            job_name = glue_job_name()
            unique_id = self.get_unique_identifier(job_name)
            return f"glue/{job_name}/{unique_id}"
        elif running_on_ecs():
            job_name = ecs_job_name()
            return f"ecs/{job_name}"
        elif running_on_docker():
            return "docker"
        else:
            return f"laptop/{getpass.getuser()}"

    def get_unique_identifier(self, job_name):
        """Get a unique identifier for the log stream."""
        return glue_job_run_id(job_name, self.boto3_session) or datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")


if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger("SageWorks")
    logger.setLevel(logging.INFO)
    cloudwatch_handler = CloudWatchHandler()
    logger.addHandler(cloudwatch_handler)
    logger.info("Test log message to CloudWatch")

    # Test the get_unique_identifier function
    print(cloudwatch_handler.get_unique_identifier("Glue_Job_1"))
