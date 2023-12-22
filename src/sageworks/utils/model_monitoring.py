"""ModelMonitoring class for monitoring SageMaker endpoints"""
import logging
import json
import pandas as pd
from sagemaker import Predictor
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
    DatasetFormat,
)

# SageWorks Imports
from sageworks.core.artifacts.endpoint_core import EndpointCore
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.s3_utils import read_s3_file


class ModelMonitoring:
    def __init__(self, endpoint_name, instance_type="ml.t3.large"):
        """ExtractModelArtifact Class
        Args:
            endpoint_name (str): Name of the endpoint to set up monitoring for
            instance_type (str): Instance type to use for monitoring. Defaults to "ml.m5.large".
                                 Other options: ml.m5.large, ml.m5.xlarge, ml.m5.2xlarge, ml.m5.4xlarge, ...
        """
        self.log = logging.getLogger("sageworks")
        self.endpoint_name = endpoint_name
        self.endpoint = EndpointCore(self.endpoint_name)

        # Initialize Class Attributes
        self.sagemaker_session = self.endpoint.sm_session
        self.sagemaker_client = self.endpoint.sm_client
        self.data_capture_path = self.endpoint.model_data_capture_path
        self.monitoring_path = self.endpoint.model_monitoring_path
        self.baseline_path = f"{self.monitoring_path}/baseline"
        self.instance_type = instance_type
        self.monitoring_schedule_name = f"{self.endpoint_name}-monitoring-schedule"
        self.monitoring_output_path = f"{self.monitoring_path}/monitoring_reports"
        self.constraints_json_file = None
        self.statistics_json_file = None

        # Initialize the DefaultModelMonitor
        self.sageworks_role = AWSAccountClamp().sageworks_execution_role_arn()
        self.model_monitor = DefaultModelMonitor(role=self.sageworks_role, instance_type=self.instance_type)

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
        self.log.important(f"Deleting old endpoint configuration: {current_endpoint_config_name}")
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

    def baseline_exists(self):
        """
        Check if baseline files exist in S3.
        """
        baseline_files = wr.s3.list_objects(self.baseline_path)
        required_files = {"constraints.json", "statistics.json"}
        return all(any(file_key.endswith(req_file) for file_key in baseline_files) for req_file in required_files)

    def create_baseline(self, baseline_s3_data_path: str, recreate: bool = False):
        """Code to create a baseline for monitoring
        Args:
            baseline_s3_data_path (str): S3 Path to the baseline data
            recreate (bool): If True, recreate the baseline even if it already exists
        Notes:
            This will write two files to the baseline_path:
            - constraints.json
            - statistics.json

            These files/locations are given to the monitoring schedule vis these two method calls
            - my_monitor.baseline_statistics()
            - my_monitor.suggested_constraints()
        """
        if not self.baseline_exists() or recreate:
            self.log.important(f"Creating baseline for {self.endpoint_name} --> {baseline_s3_data_path}")
            self.model_monitor.suggest_baseline(
                baseline_dataset=baseline_s3_data_path,
                dataset_format=DatasetFormat.csv(header=True),
                output_s3_uri=self.baseline_path,
            )
        else:
            self.log.important(f"Baseline already exists for {self.endpoint_name}")

    def check_baseline_outputs(self):
        """Code to check the outputs of the baseline
        Notes:
            This will read two files from the baseline_path:
            - constraints.json
            - statistics.json
        """
        self.constraints_json_file = f"{self.baseline_path}/constraints.json"
        self.statistics_json_file = f"{self.baseline_path}/statistics.json"

        # Read the constraint file from S3
        if not wr.s3.does_object_exist(path=self.constraints_json_file):
            self.log.warning("Constraints file does not exist in S3.")
        else:
            raw_json = read_s3_file(s3_path=self.constraints_json_file)
            constraints_data = json.loads(raw_json)
            constraints_df = pd.json_normalize(constraints_data["features"])
            print("Constraints:")
            print(constraints_df.head(20))

        # Read the statistics file from S3
        if not wr.s3.does_object_exist(path=self.statistics_json_file):
            self.log.warning("Statistics file does not exist in S3.")
        else:
            raw_json = read_s3_file(s3_path=self.statistics_json_file)
            statistics_data = json.loads(raw_json)
            statistics_df = pd.json_normalize(statistics_data["features"])
            print("Statistics:")
            print(statistics_df.head(20))

    def setup_monitoring_schedule(self, schedule: str = "hourly", recreate: bool = False):
        """
        Sets up the monitoring schedule for the model endpoint.
        Args:
            schedule (str): The schedule for the monitoring job (hourly or daily, defaults to hourly).
            recreate (bool): If True, recreate the monitoring schedule even if it already exists.
        """

        # Set up the monitoring schedule, name, and output path
        if schedule == "daily":
            schedule = CronExpressionGenerator.daily()
        else:
            schedule = CronExpressionGenerator.hourly()

        # Check if monitoring schedule already exists
        if self.monitoring_schedule_exists() and not recreate:
            return

        # Setup monitoring schedule
        self.model_monitor.create_monitoring_schedule(
            monitor_schedule_name=self.monitoring_schedule_name,
            endpoint_input=self.endpoint_name,
            output_s3_uri=self.monitoring_output_path,
            statistics=self.statistics_json_file,
            constraints=self.constraints_json_file,
            schedule_cron_expression=schedule,
        )
        self.log.important(f"Monitoring schedule created for {self.endpoint_name}.")

    def setup_alerts(self):
        """Code to set up alerts based on monitoring results"""
        pass

    def monitoring_schedule_exists(self):
        """Code to get the status of the monitoring schedule"""
        existing_schedules = self.sagemaker_client.list_monitoring_schedules(MaxResults=100).get(
            "MonitoringScheduleSummaries", []
        )
        if any(schedule["MonitoringScheduleName"] == self.monitoring_schedule_name for schedule in existing_schedules):
            self.log.info(f"Monitoring schedule already exists for {self.endpoint_name}.")
            return True
        else:
            self.log.info(f"Could not find a Monitoring schedule for {self.endpoint_name}.")
            return False


if __name__ == "__main__":
    """Exercise the ModelMonitoring class"""
    from sageworks.core.artifacts.feature_set_core import FeatureSetCore
    from sageworks.core.artifacts.model_core import ModelCore
    import awswrangler as wr

    # Set options for actually seeing the dataframe
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Create the Class and test it out
    endpoint_name = "abalone-regression-end-rt"
    my_endpoint = EndpointCore(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)
    mm = ModelMonitoring(endpoint_name)
    mm.add_data_capture()

    # Grab the FeatureSet by backtracking from the Endpoint
    model = my_endpoint.get_input()
    feature_set = ModelCore(model).get_input()
    features = FeatureSetCore(feature_set)
    table = features.get_training_view_table()

    # Create a baseline for monitoring
    my_baseline_s3_path = f"{mm.baseline_path}/baseline.csv"
    baseline_df = features.query(f"SELECT * FROM {table} where training = 1")
    wr.s3.to_csv(baseline_df, my_baseline_s3_path, index=False)

    # Now create the baseline (if it doesn't already exist)
    mm.create_baseline(my_baseline_s3_path)

    # Check the baseline outputs
    mm.check_baseline_outputs()

    # Set up the monitoring schedule (if it doesn't already exist)
    mm.setup_monitoring_schedule()

    #
    # Test the data capture by running some predictions
    #

    # Make predictions on the Endpoint
    test_df = features.query(f"SELECT * FROM {table} where training = 0")
    pred_df = my_endpoint.predict(test_df[:10])
    print(pred_df.head())

    # Check that data capture is working
    check_df = mm.check_data_capture()
    if check_df is None:
        print("No data capture files found, for a new endpoint it may take a few minutes to start capturing data")
    else:
        print("Found data capture files")
        print(check_df.head())
