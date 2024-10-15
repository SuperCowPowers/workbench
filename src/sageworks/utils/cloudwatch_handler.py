import os
import logging
from datetime import datetime
import getpass
import watchtower

# import atexit

# SageWorks imports
from sageworks.utils.execution_environment import (
    running_on_lambda,
    running_on_glue,
    running_on_ecs,
    running_on_docker,
    glue_job_name,
    ecs_job_name,
)
from sageworks.aws_service_broker.aws_session import AWSSession


class CloudWatchHandler:
    """A helper class to add a CloudWatch Logs handler to a logger"""

    def __init__(self):
        # Initialize the CloudWatch handler

        # Import ColoredFormatter here to avoid circular imports
        from sageworks.utils.sageworks_logging import ColoredFormatter

        self.boto3_session = AWSSession().boto3_session
        self.log_stream_name = self.determine_log_stream()
        self.formatter = ColoredFormatter("(%(filename)s:%(lineno)d) %(levelname)s %(message)s")
        self.cloudwatch_handler = None

    def add_cloudwatch_handler(self, log):
        """Add a CloudWatch Logs handler to the provided logger"""
        try:
            cloudwatch_client = self.boto3_session.client("logs")
            self.cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group="SageWorksLogGroup",
                stream_name=self.log_stream_name,
                boto3_client=cloudwatch_client,
            )
            self.cloudwatch_handler.setFormatter(self.formatter)
            log.addHandler(self.cloudwatch_handler)
            log.info("CloudWatch logging handler added successfully.")

            # Ensure logs are flushed after each log
            def flush_handler(record):
                self.cloudwatch_handler.flush()
                return True

            # Add a filter to flush the handler after every log
            log.addFilter(flush_handler)

        except Exception as e:
            log.error(f"Failed to set up CloudWatch Logs handler: {e}")

    # Register the flush function to be called at exit
    # atexit.register(self.on_exit_flush)
    def on_exit_flush(self):
        """Flush the CloudWatch log handler to ensure all logs are sent"""
        if hasattr(self, "cloudwatch_handler") and self.cloudwatch_handler:
            self.cloudwatch_handler.flush()
            self.cloudwatch_handler.close()

    def determine_log_stream(self):
        """Determine the log stream name based on the environment."""
        unique_id = self.get_unique_identifier()

        if running_on_lambda():
            job_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            return f"lambda/{job_name}"
        elif running_on_glue():
            job_name = glue_job_name()
            return f"glue/{job_name}/{unique_id}"
        elif running_on_ecs():
            job_name = ecs_job_name()
            return f"ecs/{job_name}"
        elif running_on_docker():
            return "docker"
        else:
            return f"laptop/{getpass.getuser()}"

    @staticmethod
    def get_unique_identifier():
        """Get a unique identifier for the log stream."""
        job_id = CloudWatchHandler.get_job_id_from_environment()
        return job_id or datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")

    @staticmethod
    def get_job_id_from_environment():
        """Try to retrieve the job ID from Glue or Lambda environment variables."""
        return os.environ.get("GLUE_JOB_ID") or os.environ.get("AWS_LAMBDA_REQUEST_ID")


if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger("SageWorks")
    cloudwatch_handler = CloudWatchHandler()
    cloudwatch_handler.add_cloudwatch_handler(logger)
