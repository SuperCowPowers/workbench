"""ModelMonitoring class for monitoring SageMaker endpoints"""
import logging
from sagemaker import Predictor
from sagemaker.model_monitor import DataCaptureConfig

# SageWorks Imports
from sageworks.artifacts.endpoints.endpoint import Endpoint


class ModelMonitoring:
    def __init__(self, endpoint_name):
        """ExtractModelArtifact Class
        Args:
            endpoint_name (str): Name of the endpoint to set up monitoring for
        """
        self.log = logging.getLogger("sageworks")
        self.endpoint_name = endpoint_name
        self.endpoint = Endpoint(self.endpoint_name)

        # Initialize Class Attributes
        self.sagemaker_session = self.endpoint.sm_session
        self.data_capture_path = self.endpoint.model_data_capture_path
        self.monitoring_path = self.endpoint.model_monitoring_path

    def add_data_capture(self, capture_percentage=100):
        """
        Add data capture configuration for the SageMaker endpoint.

        Args:
            capture_percentage (int): Percentage of data to capture. Defaults to 100.
        """

        # Check if this endpoint is a serverless endpoint
        if self.endpoint.is_serverless:
            self.log.warning(f"Data capture is not currently supported for serverless endpoints.")
            return

        # Log the data capture path
        self.log.important(f"Data capture endpoint: {self.endpoint_name} --> {self.data_capture_path}")

        # Setup data capture config
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=capture_percentage,
            destination_s3_uri=self.data_capture_path
        )

        # Create a Predictor instance
        predictor = Predictor(self.endpoint_name, sagemaker_session=self.sagemaker_session)

        # Update data capture configuration using the Predictor
        predictor.update_data_capture_config(data_capture_config=data_capture_config)

    def create_baseline(self):
        """Code to create a baseline for monitoring"""
        pass

    def setup_monitoring_schedule(self):
        """Code to set up the monitoring schedule"""
        pass

    def setup_alerts(self):
        """Code to set up alerts based on monitoring results"""
        pass

    def get_monitoring_schedule_status(self):
        """Code to get the status of the monitoring schedule"""
        pass


if __name__ == "__main__":
    """Exercise the ModelMonitoring class"""

    # Create the Class and test it out
    my_endpoint = "abalone-regression-end"
    mm = ModelMonitoring(my_endpoint)
    mm.add_data_capture()
