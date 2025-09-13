"""DataCaptureCore class for managing SageMaker endpoint data capture"""

import logging
import re
import time
from datetime import datetime
from typing import Tuple
import pandas as pd
from sagemaker import Predictor
from sagemaker.model_monitor import DataCaptureConfig
import awswrangler as wr

# Workbench Imports
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.monitor_utils import process_data_capture

# Setup logging
log = logging.getLogger("workbench")


class DataCaptureCore:
    """Manages data capture configuration and retrieval for SageMaker endpoints"""

    def __init__(self, endpoint_name: str):
        """DataCaptureCore Class

        Args:
            endpoint_name (str): Name of the endpoint to manage data capture for
        """
        self.log = logging.getLogger("workbench")
        self.endpoint_name = endpoint_name
        self.endpoint = EndpointCore(self.endpoint_name)

        # Initialize Class Attributes
        self.sagemaker_session = self.endpoint.sm_session
        self.sagemaker_client = self.endpoint.sm_client
        self.data_capture_path = self.endpoint.endpoint_data_capture_path
        self.workbench_role_arn = AWSAccountClamp().aws_session.get_workbench_execution_role_arn()

    def summary(self) -> dict:
        """Return the summary of data capture configuration

        Returns:
            dict: Summary of data capture status
        """
        if self.endpoint.is_serverless():
            return {"endpoint_type": "serverless", "data_capture": "not supported"}
        else:
            return {
                "endpoint_type": "realtime",
                "data_capture_enabled": self.is_enabled(),
                "capture_percentage": self.capture_percentage(),
                "capture_modes": self.capture_modes() if self.is_enabled() else [],
                "data_capture_path": self.data_capture_path if self.is_enabled() else None,
            }

    def enable(self, capture_percentage=100, capture_options=None, force_redeploy=False):
        """
        Enable data capture for the SageMaker endpoint.

        Args:
            capture_percentage (int): Percentage of data to capture. Defaults to 100.
            capture_options (list): List of what to capture - ["REQUEST"], ["RESPONSE"], or ["REQUEST", "RESPONSE"].
                                    Defaults to ["REQUEST", "RESPONSE"] to capture both.
            force_redeploy (bool): If True, force redeployment even if data capture is already enabled.
        """
        # Early returns for cases where we can't/don't need to add data capture
        if self.endpoint.is_serverless():
            self.log.warning("Data capture is not supported for serverless endpoints.")
            return

        # Default to capturing both if not specified
        if capture_options is None:
            capture_options = ["REQUEST", "RESPONSE"]

        # Validate capture_options
        valid_options = {"REQUEST", "RESPONSE"}
        if not all(opt in valid_options for opt in capture_options):
            self.log.error("Invalid capture_options. Must be a list containing 'REQUEST' and/or 'RESPONSE'")
            return

        if self.is_enabled() and not force_redeploy:
            self.log.important(f"Data capture already configured for {self.endpoint_name}.")
            return

        # Get the current endpoint configuration name for later deletion
        current_endpoint_config_name = self.endpoint.endpoint_config_name()

        # Log the data capture operation
        self.log.important(f"Enabling Data Capture for {self.endpoint_name} --> {self.data_capture_path}")
        self.log.important(f"Capturing: {', '.join(capture_options)} at {capture_percentage}% sampling")
        self.log.important("This will redeploy the endpoint...")

        # Create and apply the data capture configuration
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=capture_percentage,
            destination_s3_uri=self.data_capture_path,
            capture_options=capture_options,
        )

        # Update endpoint with the new capture configuration
        Predictor(self.endpoint_name, sagemaker_session=self.sagemaker_session).update_data_capture_config(
            data_capture_config=data_capture_config
        )

        # Clean up old endpoint configuration
        try:
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=current_endpoint_config_name)
            self.log.info(f"Deleted old endpoint configuration: {current_endpoint_config_name}")
        except Exception as e:
            self.log.warning(f"Could not delete old endpoint configuration {current_endpoint_config_name}: {e}")

    def disable(self):
        """
        Disable data capture for the SageMaker endpoint.
        """
        # Early return if data capture isn't configured
        if not self.is_enabled():
            self.log.important(f"Data capture is not currently enabled for {self.endpoint_name}.")
            return

        # Get the current endpoint configuration name for later deletion
        current_endpoint_config_name = self.endpoint.endpoint_config_name()

        # Log the operation
        self.log.important(f"Disabling Data Capture for {self.endpoint_name}")
        self.log.important("This normally redeploys the endpoint...")

        # Create a configuration with capture disabled
        data_capture_config = DataCaptureConfig(enable_capture=False, destination_s3_uri=self.data_capture_path)

        # Update endpoint with the new configuration
        Predictor(self.endpoint_name, sagemaker_session=self.sagemaker_session).update_data_capture_config(
            data_capture_config=data_capture_config
        )

        # Clean up old endpoint configuration
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=current_endpoint_config_name)

    def is_enabled(self) -> bool:
        """
        Check if data capture is enabled on the endpoint.

        Returns:
            bool: True if data capture is enabled, False otherwise.
        """
        try:
            endpoint_config_name = self.endpoint.endpoint_config_name()
            endpoint_config = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            data_capture_config = endpoint_config.get("DataCaptureConfig", {})

            # Check if data capture is enabled
            is_enabled = data_capture_config.get("EnableCapture", False)
            return is_enabled
        except Exception as e:
            self.log.error(f"Error checking data capture configuration: {e}")
            return False

    def capture_percentage(self) -> int:
        """
        Get the data capture percentage from the endpoint configuration.

        Returns:
            int: Data capture percentage if enabled, None otherwise.
        """
        try:
            endpoint_config_name = self.endpoint.endpoint_config_name()
            endpoint_config = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            data_capture_config = endpoint_config.get("DataCaptureConfig", {})

            # Check if data capture is enabled and return the percentage
            if data_capture_config.get("EnableCapture", False):
                return data_capture_config.get("InitialSamplingPercentage", 0)
            else:
                return None
        except Exception as e:
            self.log.error(f"Error checking data capture percentage: {e}")
            return None

    def get_config(self) -> dict:
        """
        Returns the complete data capture configuration from the endpoint config.

        Returns:
            dict: Complete DataCaptureConfig from AWS, or None if not configured
        """
        config_name = self.endpoint.endpoint_config_name()
        response = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)
        data_capture_config = response.get("DataCaptureConfig")
        if not data_capture_config:
            self.log.error(f"No data capture configuration found for endpoint config {config_name}")
            return None
        return data_capture_config

    def capture_modes(self) -> list:
        """Get the current capture modes (REQUEST/RESPONSE)"""
        if not self.is_enabled():
            return []

        config = self.get_config()
        if not config:
            return []

        capture_options = config.get("CaptureOptions", [])
        modes = [opt.get("CaptureMode") for opt in capture_options]
        return ["REQUEST" if m == "Input" else "RESPONSE" for m in modes if m]

    def get_captured_data(self, from_date: str = None, add_timestamp: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read and process captured data from S3.

        Args:
            from_date (str, optional): Only process files from this date onwards (YYYY-MM-DD format).
                                       Defaults to None to process all files.
            add_timestamp (bool, optional): Whether to add a timestamp column to the DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed input and output DataFrames.
        """
        files = wr.s3.list_objects(self.data_capture_path)
        if not files:
            self.log.warning(f"No data capture files found in {self.data_capture_path}.")
            return pd.DataFrame(), pd.DataFrame()

        # Filter by date if specified
        if from_date:
            from_date_obj = datetime.strptime(from_date, "%Y-%m-%d").date()
            files = [f for f in files if self._file_date_filter(f, from_date_obj)]
            self.log.info(f"Processing {len(files)} files from {from_date} onwards.")
        else:
            self.log.info(f"Processing all {len(files)} files...")

        # Check if any files remain after filtering
        if not files:
            self.log.info("No files to process after date filtering.")
            return pd.DataFrame(), pd.DataFrame()

        # Sort files by name (assumed to include timestamp)
        files.sort()

        # Get all timestamps in one batch if needed
        timestamps = {}
        if add_timestamp:
            # Batch describe operation - much more efficient than per-file calls
            timestamps = wr.s3.describe_objects(path=files)

        # Process files using concurrent.futures
        start_time = time.time()

        def process_single_file(file_path):
            """Process a single file and return input/output DataFrames."""
            try:
                log.debug(f"Processing file: {file_path}...")
                df = wr.s3.read_json(path=file_path, lines=True)
                if not df.empty:
                    input_df, output_df = process_data_capture(df)
                    if add_timestamp and file_path in timestamps:
                        output_df["timestamp"] = timestamps[file_path]["LastModified"]
                    return input_df, output_df
                return pd.DataFrame(), pd.DataFrame()
            except Exception as e:
                self.log.warning(f"Error processing {file_path}: {e}")
                return pd.DataFrame(), pd.DataFrame()

        # Use ThreadPoolExecutor for I/O-bound operations
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(32, len(files))  # Cap at 32 threads or number of files

        all_input_dfs, all_output_dfs = [], []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, file_path) for file_path in files]
            for future in futures:
                input_df, output_df = future.result()
                if not input_df.empty:
                    all_input_dfs.append(input_df)
                if not output_df.empty:
                    all_output_dfs.append(output_df)

        if not all_input_dfs:
            self.log.warning("No valid data was processed.")
            return pd.DataFrame(), pd.DataFrame()

        input_df = pd.concat(all_input_dfs, ignore_index=True)
        output_df = pd.concat(all_output_dfs, ignore_index=True)

        elapsed_time = time.time() - start_time
        self.log.info(f"Processed {len(files)} files in {elapsed_time:.2f} seconds.")
        return input_df, output_df

    def _file_date_filter(self, file_path, from_date_obj):
        """Extract date from S3 path and compare with from_date."""
        try:
            # Match YYYY/MM/DD pattern in the path
            date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", file_path)
            if date_match:
                year, month, day = date_match.groups()
                file_date = datetime(int(year), int(month), int(day)).date()
                return file_date >= from_date_obj
            return False  # No date pattern found
        except ValueError:
            return False

    def __repr__(self) -> str:
        """String representation of this DataCaptureCore object

        Returns:
            str: String representation of this DataCaptureCore object
        """
        summary_dict = self.summary()
        summary_items = [f"  {repr(key)}: {repr(value)}" for key, value in summary_dict.items()]
        summary_str = f"{self.__class__.__name__}: {self.endpoint_name}\n" + ",\n".join(summary_items)
        return summary_str


# Test function for the class
if __name__ == "__main__":
    """Exercise the MonitorCore class"""
    from pprint import pprint

    # Set options for actually seeing the dataframe
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Create the Class and test it out
    endpoint_name = "abalone-regression-rt"
    my_endpoint = EndpointCore(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)
    dc = my_endpoint.data_capture()

    # Check the summary of the data capture class
    pprint(dc.summary())

    # Enable data capture on the endpoint
    # dc.enable(force_redeploy=True)
    my_endpoint.enable_data_capture()

    # Test the data capture by running some predictions
    # pred_df = my_endpoint.auto_inference()
    # print(pred_df.head())

    # Check that data capture is working
    input_df, output_df = dc.get_captured_data(from_date="2025-09-01")
    if input_df.empty and output_df.empty:
        print("No data capture files found, for a new endpoint it may take a few minutes to start capturing data")
    else:
        print("Found data capture files")
        print("Input")
        print(input_df.head())
        print("Output")
        print(output_df.head())
