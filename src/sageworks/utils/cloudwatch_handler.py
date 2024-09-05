import os
import logging
from datetime import datetime
import getpass
import watchtower
import atexit  # Import atexit for handling cleanup on exit

# SageWorks imports
from sageworks.utils.docker_utils import running_on_docker


class CloudWatchHandler:
    """A helper class to add a CloudWatch Logs handler to a logger"""

    def __init__(self):
        # Initialize the CloudWatch handler
        try:
            # Import AWSAccountClamp here to avoid circular imports
            from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
            from sageworks.utils.sageworks_logging import ColoredFormatter

            self.account_clamp = AWSAccountClamp()
            self.boto3_session = self.account_clamp.boto_session()
            self.log_stream_name = self.determine_log_stream()
            self.formatter = ColoredFormatter("(%(filename)s:%(lineno)d) %(levelname)s %(message)s")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize CloudWatchHandler: {e}")

    def add_cloudwatch_handler(self, log):
        """Add a CloudWatch Logs handler to the provided logger"""
        try:
            cloudwatch_client = self.boto3_session.client("logs")
            self.cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group="SageWorksLogGroup",
                stream_name=self.log_stream_name,
                boto3_client=cloudwatch_client,
                send_interval=5,
            )
            self.cloudwatch_handler.setFormatter(self.formatter)
            log.addHandler(self.cloudwatch_handler)
            log.info("CloudWatch logging handler added successfully.")

            # Register the flush function to be called at exit
            atexit.register(self.flush_handler)

        except Exception as e:
            log.error(f"Failed to set up CloudWatch Logs handler: {e}")

    def flush_handler(self):
        """Flush the CloudWatch log handler to ensure all logs are sent"""
        if hasattr(self, "cloudwatch_handler") and self.cloudwatch_handler:
            self.cloudwatch_handler.flush()

    def determine_log_stream(self):
        """Determine the log stream name based on the environment."""
        unique_id = self.get_unique_identifier()

        if self.running_on_lambda():
            job_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            return f"lambda/{job_name}"
        elif self.running_on_glue():
            job_name = os.environ.get("GLUE_JOB_NAME", "unknown")
            return f"glue/{job_name}/{unique_id}"
        elif running_on_docker():
            job_name = os.environ.get("ECS_SERVICE_NAME") or os.environ.get("HOSTNAME", "unknown")
            return f"docker/{job_name}"
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

    @staticmethod
    def running_on_lambda():
        """Check if running in AWS Lambda."""
        return "AWS_LAMBDA_FUNCTION_NAME" in os.environ

    @staticmethod
    def running_on_glue():
        """Check if running in AWS Glue."""
        return "GLUE_JOB_NAME" in os.environ


if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger("SageWorks")
    cloudwatch_handler = CloudWatchHandler()
    cloudwatch_handler.add_cloudwatch_handler(logger)
