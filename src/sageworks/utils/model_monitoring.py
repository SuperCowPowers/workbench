"""ModelMonitoring class for monitoring SageMaker endpoints"""
import logging
import pandas as pd
import awswrangler as wr
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
        self.sagemaker_client = self.endpoint.sm_client
        self.data_capture_path = self.endpoint.model_data_capture_path
        self.monitoring_path = self.endpoint.model_monitoring_path

    def add_data_capture(self, capture_percentage=100):
        """
        Add data capture configuration for the SageMaker endpoint.

        Args:
            capture_percentage (int): Percentage of data to capture. Defaults to 100.
        """

        # Check if this endpoint is a serverless endpoint
        if self.endpoint.is_serverless():
            self.log.warning("Data capture is not currently supported for serverless endpoints.")
            return

        # Check if the endpoint already has data capture configured
        if self.is_data_capture_configured(capture_percentage):
            self.log.important(f"Data capture {capture_percentage} already configured for {self.endpoint_name}.")
            return

        # Get the current endpoint configuration name
        current_endpoint_config_name = self.endpoint.endpoint_config_name()

        # Log the data capture path
        self.log.important(f"Adding Data Capture to {self.endpoint_name} --> {self.data_capture_path}")
        self.log.important("This normally redeploys the endpoint...")

        # Setup data capture config
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=capture_percentage,
            destination_s3_uri=self.data_capture_path,
            capture_options=["Input", "Output"],
            csv_content_types=["text/csv"],
        )

        # Create a Predictor instance and update data capture configuration
        predictor = Predictor(self.endpoint_name, sagemaker_session=self.sagemaker_session)
        predictor.update_data_capture_config(data_capture_config=data_capture_config)

        # Delete the old endpoint configuration
        self.log.warning(f"Deleting old endpoint configuration: {current_endpoint_config_name}")
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=current_endpoint_config_name)

    def is_data_capture_configured(self, capture_percentage):
        """
        Check if data capture is already configured on the endpoint.
        Args:
            capture_percentage (int): Expected data capture percentage.
        Returns:
            bool: True if data capture is already configured, False otherwise.
        """
        try:
            endpoint_config_name = self.endpoint.endpoint_config_name()
            endpoint_config = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            data_capture_config = endpoint_config.get("DataCaptureConfig", {})

            # Check if data capture is enabled and the percentage matches
            is_enabled = data_capture_config.get("EnableCapture", False)
            current_percentage = data_capture_config.get("InitialSamplingPercentage", 0)
            return is_enabled and current_percentage == capture_percentage
        except Exception as e:
            self.log.error(f"Error checking data capture configuration: {e}")
            return False

    def check_data_capture(self):
        """
        Check the data capture by reading the captured data from S3.
        Returns:
            DataFrame: The captured data as a Pandas DataFrame.
        """
        # List files in the specified S3 path
        files = wr.s3.list_objects(self.data_capture_path)

        if files:
            print(f"Found {len(files)} files in {self.data_capture_path}. Reading the most recent file.")

            # Read the most recent file into a DataFrame
            df = wr.s3.read_json(path=files[-1], lines=True)  # Reads the last file assuming it's the most recent one

            # Process the captured data
            processed_df = self.process_captured_data(df)
            return processed_df
        else:
            print(f"No data capture files found in {self.data_capture_path}.")
            return None

    @staticmethod
    def process_captured_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the captured data DataFrame to extract and flatten the nested data.
        Args:
            df (DataFrame): DataFrame with captured data.
        Returns:
            DataFrame: Flattened and processed DataFrame.
        """
        processed_records = []

        for _, row in df.iterrows():
            # Extract data from captureData dictionary
            capture_data = row["captureData"]
            input_data = capture_data["endpointInput"]
            output_data = capture_data["endpointOutput"]

            # Process input and output data as needed
            # Here, we're extracting specific fields, but you might need to adjust this based on your data structure
            record = {
                "input_content_type": input_data.get("observedContentType"),
                "input_encoding": input_data.get("encoding"),
                "input": input_data.get("data"),
                "output_content_type": output_data.get("observedContentType"),
                "output_encoding": output_data.get("encoding"),
                "output": output_data.get("data"),
            }
            processed_records.append(record)

        return pd.DataFrame(processed_records)

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
    from sageworks.artifacts.feature_sets.feature_set import FeatureSet
    from sageworks.artifacts.models.model import Model

    # Set options for actually seeing the dataframe
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Create the Class and test it out
    endpoint_name = "aqsol-regression-end"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)
    mm = ModelMonitoring(endpoint_name)
    mm.add_data_capture()

    #
    # Test the data capture by running some predictions
    #

    # Grab the FeatureSet by backtracking from the Endpoint
    model = my_endpoint.get_input()
    feature_set = Model(model).get_input()
    features = FeatureSet(feature_set)
    table = features.get_training_view_table()
    test_df = features.query(f"SELECT * FROM {table} where training = 0")

    # Drop some columns
    test_df.drop(["write_time", "api_invocation_time", "is_deleted"], axis=1, inplace=True)

    # Make predictions on the Endpoint
    pred_df = my_endpoint.predict(test_df[:10])
    print(pred_df.head())

    # Check that data capture is working
    check_df = mm.check_data_capture()
    if check_df is None:
        print("No data capture files found, for a new endpoint it may take a few minutes to start capturing data")
    else:
        print("Found data capture files")
        print(check_df.head())
