import os
import sys
import logging
from datetime import datetime
import getpass
import watchtower

# SageWorks imports
from sageworks.utils.docker_utils import running_on_docker


class CloudWatchHandler(logging.Handler):
    """A custom CloudWatch Logs handler for SageWorks"""

    def __init__(self):
        super().__init__()

        # Initialize the CloudWatch handler
        try:
            # Import AWSAccountClamp here to avoid circular imports
            from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
            from sageworks.utils.sageworks_logging import ColoredFormatter

            self.account_clamp = AWSAccountClamp()
            self.boto3_session = self.account_clamp.boto_session()
            self.log_stream_name = self.determine_log_stream()
            self.formatter = ColoredFormatter("(%(filename)s:%(lineno)d) %(levelname)s %(message)s")
            self.setFormatter(self.formatter)
            cloudwatch_client = self.boto3_session.client("logs")
            self.cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group="SageWorksLogGroup",
                stream_name=self.log_stream_name,
                boto3_client=cloudwatch_client,
            )
        except Exception as e:
            self.cloudwatch_handler = None
            raise e

    def emit(self, record):
        """Emit a log record to CloudWatch."""
        if self.cloudwatch_handler:
            msg = self.format(record)
            record.msg = msg
            self.cloudwatch_handler.emit(record)

    @staticmethod
    def get_executable_name(argv):
        try:
            script_path = argv[0]
            base_name = os.path.basename(script_path)
            return os.path.splitext(base_name)[0]
        except Exception:
            return None

    def determine_log_stream(self):
        """Determine the log stream name based on the environment."""
        executable_name = self.get_executable_name(sys.argv)
        unique_id = self.get_unique_identifier()

        if self.running_on_lambda():
            job_name = executable_name or os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            return f"lambda/{job_name}"
        elif self.running_on_glue():
            job_name = executable_name or os.environ.get("GLUE_JOB_NAME", "unknown")
            return f"glue/{job_name}/{unique_id}"
        elif running_on_docker():
            job_name = executable_name or os.environ.get("SERVICE_NAME", "unknown")
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
    logger.addHandler(CloudWatchHandler())
