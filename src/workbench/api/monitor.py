"""Monitor: Manages AWS Endpoint Monitor creation and deployment.
Endpoints Monitors are set up and provisioned for deployment into AWS.
Monitors can be viewed in the AWS Sagemaker interfaces or in the Workbench
Dashboard UI, which provides additional monitor details and performance metrics"""

import pandas as pd
from typing import Union

# Workbench Imports
from workbench.core.artifacts.monitor_core import MonitorCore


class Monitor(MonitorCore):
    """Monitor: Workbench Monitor API Class

    Common Usage:
       ```
       mon = Endpoint(name).get_monitor()  # Pull from endpoint OR
       mon = Monitor(name)                 # Create using Endpoint Name
       mon.summary()
       mon.details()

       # One time setup methods
       mon.enable_data_capture()
       mon.create_baseline()
       mon.create_monitoring_schedule()

       # Pull information from the monitor
       baseline_df = mon.get_baseline()
       constraints_df = mon.get_constraints()
       stats_df = mon.get_statistics()
       input_df, output_df = mon.get_captured_data()
       ```
    """

    def summary(self) -> dict:
        """Monitor Summary

        Returns:
            dict: A dictionary of summary information about the Monitor
        """
        return super().summary()

    def details(self) -> dict:
        """Monitor Details

        Returns:
            dict: A dictionary of details about the Monitor
        """
        return super().details()

    def enable_data_capture(self, capture_percentage=100):
        """
        Enable data capture configuration for this Monitor/endpoint.

        Args:
            capture_percentage (int): Percentage of data to capture. Defaults to 100.
        """
        super().enable_data_capture(capture_percentage)

    def create_baseline(self, recreate: bool = False):
        """Code to create a baseline for monitoring

        Args:
            recreate (bool): If True, recreate the baseline even if it already exists

        Notes:
            This will create/write three files to the baseline_dir:
            - baseline.csv
            - constraints.json
            - statistics.json
        """
        super().create_baseline(recreate)

    def create_monitoring_schedule(self, schedule: str = "hourly"):
        """
        Sets up the monitoring schedule for the model endpoint.

        Args:
            schedule (str): The schedule for the monitoring job (hourly or daily, defaults to hourly).
        """
        super().create_monitoring_schedule(schedule)

    def get_captured_data(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Get the latest data capture input and output from S3.

        Returns:
            DataFrame (input), DataFrame(output): Flattened and processed DataFrames for input and output data.
        """
        return super().get_captured_data()

    def get_baseline(self) -> Union[pd.DataFrame, None]:
        """Code to get the baseline CSV from the S3 baseline directory

        Returns:
            pd.DataFrame: The baseline CSV as a DataFrame (None if it doesn't exist)
        """
        return super().get_baseline()

    def get_constraints(self) -> Union[pd.DataFrame, None]:
        """Code to get the constraints from the baseline

        Returns:
           pd.DataFrame: The constraints from the baseline (constraints.json) (None if it doesn't exist)
        """
        return super().get_constraints()

    def get_statistics(self) -> Union[pd.DataFrame, None]:
        """Code to get the statistics from the baseline

        Returns:
            pd.DataFrame: The statistics from the baseline (statistics.json) (None if it doesn't exist)
        """
        return super().get_statistics()


if __name__ == "__main__":
    """Exercise the MonitorCore class"""
    from pprint import pprint
    from workbench.api.endpoint import Endpoint

    # Set options for actually seeing the dataframe
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Create the Class and test it out
    endpoint_name = "abalone-regression-rt"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)
    mm = Monitor(endpoint_name)

    # Check the summary and details of the monitoring class
    pprint(mm.summary())
    pprint(mm.details())

    # Enable data capture for this endpoint
    mm.enable_data_capture()

    # Create a baseline for monitoring
    mm.create_baseline()

    # Set up the monitoring schedule (if it doesn't already exist)
    mm.create_monitoring_schedule()

    # Check the monitoring outputs
    print("\nBaseline outputs...")
    base_df = mm.get_baseline()
    print(base_df.head())

    print("\nConstraints...")
    pprint(mm.get_constraints())

    print("\nStatistics...")
    print(mm.get_statistics())

    # Get the latest data capture
    input_df, output_df = mm.get_captured_data()
    print(input_df.head())
    print(output_df.head())
