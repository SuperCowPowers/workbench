"""MonitorCore class for monitoring SageMaker endpoints"""

import logging
import json
from typing import Union, Tuple
import pandas as pd
from sagemaker import Predictor
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
    DatasetFormat,
)
import awswrangler as wr

# Workbench Imports
from workbench.core.artifacts.endpoint_core import EndpointCore
from workbench.api import Model, FeatureSet
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.s3_utils import read_from_s3, write_to_s3
from workbench.utils.datetime_utils import datetime_string
from workbench.utils.monitor_utils import (
    process_data_capture,
    get_monitor_json_data,
    parse_monitoring_results,
    preprocessing_script,
)

# Note: This resource might come in handy when doing code refactoring
# https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production/blob/master/06-monitoring.ipynb
# https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-pre-and-post-processing.html
# https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker_model_monitor/introduction/SageMaker-ModelMonitoring.ipynb


class MonitorCore:
    def __init__(self, endpoint_name, instance_type="ml.m5.large"):
        """MonitorCore Class
        Args:
            endpoint_name (str): Name of the endpoint to set up monitoring for
            instance_type (str): Instance type to use for monitoring. Defaults to "ml.t3.medium".
        """
        self.log = logging.getLogger("workbench")
        self.endpoint_name = endpoint_name
        self.endpoint = EndpointCore(self.endpoint_name)

        # Initialize Class Attributes
        self.sagemaker_session = self.endpoint.sm_session
        self.sagemaker_client = self.endpoint.sm_client
        self.data_capture_path = self.endpoint.endpoint_data_capture_path
        self.monitoring_path = self.endpoint.endpoint_monitoring_path
        self.monitoring_schedule_name = f"{self.endpoint_name}-monitoring-schedule"
        self.baseline_dir = f"{self.monitoring_path}/baseline"
        self.baseline_csv_file = f"{self.baseline_dir}/baseline.csv"
        self.constraints_json_file = f"{self.baseline_dir}/constraints.json"
        self.statistics_json_file = f"{self.baseline_dir}/statistics.json"
        self.preprocessing_script_file = f"{self.monitoring_path}/preprocessor.py"
        self.workbench_role_arn = AWSAccountClamp().aws_session.get_workbench_execution_role_arn()
        self.instance_type = instance_type

        # Check if a monitoring schedule already exists for this endpoint
        existing_schedule = self.monitoring_schedule_exists()

        if existing_schedule:
            # If a schedule exists, attach to it
            self.model_monitor = DefaultModelMonitor.attach(
                monitor_schedule_name=self.monitoring_schedule_name, sagemaker_session=self.sagemaker_session
            )
            self.log.info(f"Attached to existing monitoring schedule for {self.endpoint_name}")
        else:
            # Create a new model monitor
            self.model_monitor = DefaultModelMonitor(
                role=self.workbench_role_arn, instance_type=self.instance_type, sagemaker_session=self.sagemaker_session
            )
            self.log.info(f"Initialized new model monitor for {self.endpoint_name}")

    def summary(self) -> dict:
        """Return the summary of information about the endpoint monitor

        Returns:
            dict: Summary of information about the endpoint monitor
        """
        if self.endpoint.is_serverless():
            return {
                "endpoint_type": "serverless",
                "data_capture": "not supported",
                "baseline": "not supported",
                "monitoring_schedule": "not supported",
            }
        else:
            summary = {
                "endpoint_type": "realtime",
                "data_capture": self.data_capture_enabled(),
                "capture_percent": self.data_capture_percent(),
                "baseline": self.baseline_exists(),
                "monitoring_schedule": self.monitoring_schedule_exists(),
                "preprocessing": self.preprocessing_exists(),
            }
            return summary

    def details(self) -> dict:
        """Return the details of the monitoring for the endpoint

        Returns:
            dict: The monitoring details for the endpoint
        """
        result = self.summary()
        info = {
            "data_capture_path": self.data_capture_path if self.data_capture_enabled() else None,
            "preprocessing_script_file": self.preprocessing_script_file if self.preprocessing_exists() else None,
            "monitoring_schedule_status": "Not Scheduled",
        }
        result.update(info)

        if self.baseline_exists():
            result.update(
                {
                    "baseline_csv_file": self.baseline_csv_file,
                    "baseline_constraints_json_file": self.constraints_json_file,
                    "baseline_statistics_json_file": self.statistics_json_file,
                }
            )

        if self.monitoring_schedule_exists():
            schedule_details = self.sagemaker_client.describe_monitoring_schedule(
                MonitoringScheduleName=self.monitoring_schedule_name
            )

            result.update(
                {
                    "monitoring_schedule_name": schedule_details.get("MonitoringScheduleName"),
                    "monitoring_schedule_status": schedule_details.get("MonitoringScheduleStatus"),
                    "monitoring_path": self.monitoring_path,
                    "creation_time": datetime_string(schedule_details.get("CreationTime")),
                }
            )

            last_run = schedule_details.get("LastMonitoringExecutionSummary", {})
            if last_run:

                # If no inference was run since the last monitoring schedule, the
                # status will be "Failed" with reason "Job inputs had no data",
                # so we check for that and set the status to "No New Data"
                status = last_run.get("MonitoringExecutionStatus")
                reason = last_run.get("FailureReason")
                if status == "Failed" and reason == "Job inputs had no data":
                    status = reason = "No New Data"
                result.update(
                    {
                        "last_run_status": status,
                        "last_run_time": datetime_string(last_run.get("ScheduledTime")),
                        "failure_reason": reason,
                    }
                )

        return result

    def enable_data_capture(self, capture_percentage=100):
        """
        Enable data capture for the SageMaker endpoint.

        Args:
            capture_percentage (int): Percentage of data to capture. Defaults to 100.
        """
        # Early returns for cases where we can't/don't need to add data capture
        if self.endpoint.is_serverless():
            self.log.warning("Data capture is not supported for serverless endpoints.")
            return

        if self.data_capture_enabled():
            self.log.important(f"Data capture already configured for {self.endpoint_name}.")
            return

        # Get the current endpoint configuration name for later deletion
        current_endpoint_config_name = self.endpoint.endpoint_config_name()

        # Log the data capture operation
        self.log.important(f"Enabling Data Capture for {self.endpoint_name} --> {self.data_capture_path}")
        self.log.important("This normally redeploys the endpoint...")

        # Create and apply the data capture configuration
        data_capture_config = DataCaptureConfig(
            enable_capture=True,  # Required parameter
            sampling_percentage=capture_percentage,
            destination_s3_uri=self.data_capture_path,
        )

        # Update endpoint with the new capture configuration
        Predictor(self.endpoint_name, sagemaker_session=self.sagemaker_session).update_data_capture_config(
            data_capture_config=data_capture_config
        )

        # Clean up old endpoint configuration
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=current_endpoint_config_name)

    def disable_data_capture(self):
        """
        Disable data capture for the SageMaker endpoint.
        """
        # Early return if data capture isn't configured
        if not self.data_capture_enabled():
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

    def data_capture_enabled(self):
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
            return is_enabled
        except Exception as e:
            self.log.error(f"Error checking data capture configuration: {e}")
            return False

    def data_capture_percent(self):
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

    def get_captured_data(self, max_files=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read and process captured data from S3.

        Args:
            max_files (int, optional): Maximum number of files to process.
                                      Defaults to 1 (most recent only).
                                      Set to None to process all files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed input and output DataFrames.
        """
        # List files in the specified S3 path
        files = wr.s3.list_objects(self.data_capture_path)
        if not files:
            self.log.warning(f"No data capture files found in {self.data_capture_path}.")
            return pd.DataFrame(), pd.DataFrame()

        self.log.info(f"Found {len(files)} files in {self.data_capture_path}.")

        # Sort files by timestamp (assuming the naming convention includes timestamp)
        files.sort()

        # Select files to process
        if max_files is None:
            files_to_process = files
            self.log.info(f"Processing all {len(files)} files.")
        else:
            files_to_process = files[-max_files:] if files else []
            self.log.info(f"Processing the {len(files_to_process)} most recent file(s).")

        # Process each file
        all_input_dfs = []
        all_output_dfs = []
        for file_path in files_to_process:
            try:
                # Read the JSON lines file
                df = wr.s3.read_json(path=file_path, lines=True)

                if not df.empty:
                    input_df, output_df = process_data_capture(df)
                    all_input_dfs.append(input_df)
                    all_output_dfs.append(output_df)
            except Exception as e:
                self.log.warning(f"Error processing file {file_path}: {e}")

        # Combine all DataFrames
        if not all_input_dfs or not all_output_dfs:
            self.log.warning("No valid data was processed from the captured files.")
            return pd.DataFrame(), pd.DataFrame()

        return pd.concat(all_input_dfs, ignore_index=True), pd.concat(all_output_dfs, ignore_index=True)

    def baseline_exists(self) -> bool:
        """
        Check if baseline files exist in S3.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        files = [self.baseline_csv_file, self.constraints_json_file, self.statistics_json_file]
        return all(wr.s3.does_object_exist(file) for file in files)

    def preprocessing_exists(self) -> bool:
        """
        Check if preprocessing script exists in S3.

        Returns:
            bool: True if the preprocessing script exists, False otherwise.
        """
        return wr.s3.does_object_exist(self.preprocessing_script_file)

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
        # Check if this endpoint is a serverless endpoint
        if self.endpoint.is_serverless():
            self.log.warning("Model monitoring doesn't work for serverless endpoints, skipping...")
            return

        if self.baseline_exists() and not recreate:
            self.log.info(f"Baseline already exists for {self.endpoint_name}.")
            self.log.info("If you want to recreate it, set recreate=True.")
            return

        # Create a baseline for monitoring (all rows from the FeatureSet)
        model = Model(self.endpoint.get_input())
        fs = FeatureSet(model.get_input())
        baseline_df = fs.pull_dataframe()

        # We only want the model features for our baseline
        features = model.features()
        baseline_df = baseline_df[features]

        # Sort the columns to ensure consistent ordering (AWS/Spark needs this)
        baseline_df = baseline_df[sorted(baseline_df.columns)]

        # Write the baseline to S3
        wr.s3.to_csv(baseline_df, self.baseline_csv_file, index=False)

        # Create the baseline files (constraints.json and statistics.json)
        self.log.important(f"Creating baseline files for {self.endpoint_name} --> {self.baseline_dir}")
        self.model_monitor.suggest_baseline(
            baseline_dataset=self.baseline_csv_file,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=self.baseline_dir,
        )

    def get_baseline(self) -> Union[pd.DataFrame, None]:
        """Code to get the baseline CSV from the S3 baseline directory

        Returns:
            pd.DataFrame: The baseline CSV as a DataFrame (None if it doesn't exist)
        """
        # Read the monitoring data from S3
        if not wr.s3.does_object_exist(path=self.baseline_csv_file):
            self.log.warning("baseline.csv data does not exist in S3.")
            return None
        else:
            return wr.s3.read_csv(self.baseline_csv_file)

    def get_constraints(self) -> dict:
        """Code to get the constraints from the baseline

        Returns:
            dict: The constraints associated with the monitor (constraints.json) (None if it doesn't exist)
        """
        return get_monitor_json_data(self.constraints_json_file)

    def get_statistics(self) -> Union[pd.DataFrame, None]:
        """Code to get the statistics from the baseline

        Returns:
            pd.DataFrame: The statistics from the baseline (statistics.json) (None if it doesn't exist)
        """
        # Use the utility function
        return get_monitor_json_data(self.statistics_json_file)

    def update_constraints(self, constraints_updates):
        """Update the constraints file with custom constraints or monitoring config

        Args:
            constraints_updates (dict): Dictionary of updates to apply to the constraints file.
                - If key is "monitoring_config", updates the monitoring configuration
                - Otherwise, treated as feature name with constraint updates

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.baseline_exists():
            self.log.warning("Cannot update constraints without a baseline")
            return False

        try:
            # Read constraints file from S3
            raw_json = read_from_s3(self.constraints_json_file)
            constraints = json.loads(raw_json)

            # Handle each update key
            for key, updates in constraints_updates.items():
                if key == "monitoring_config":
                    # Update monitoring config
                    if "monitoring_config" not in constraints:
                        constraints["monitoring_config"] = {}
                    constraints["monitoring_config"].update(updates)
                else:
                    # Update feature constraints
                    feature_found = False
                    for feature in constraints.get("features", []):
                        if feature["name"] == key:
                            feature.update(updates)
                            feature_found = True
                            break

                    if not feature_found:
                        self.log.warning(f"Feature {key} not found in constraints")

            # Write updated constraints back to S3
            write_to_s3(json.dumps(constraints, indent=2), self.constraints_json_file)
            self.log.important(f"Updated constraints at {self.constraints_json_file}")
            return True
        except Exception as e:
            self.log.error(f"Error updating constraints: {e}")
            return False

    def create_monitoring_schedule(self, schedule: str = "hourly"):
        """Sets up the monitoring schedule for the model endpoint.

        Args:
            schedule (str): The schedule for the monitoring job (hourly or daily, defaults to hourly).
        """
        # Check if this endpoint is a serverless endpoint
        if self.endpoint.is_serverless():
            self.log.warning("Monitoring Schedule is not currently supported for serverless endpoints.")
            return

        # Set up the monitoring schedule, name, and output path
        if schedule == "daily":
            schedule = CronExpressionGenerator.daily()
        else:
            schedule = CronExpressionGenerator.hourly()

        # Check if the baseline exists
        if not self.baseline_exists():
            self.log.warning(f"Baseline does not exist for {self.endpoint_name}. Call create_baseline() first...")
            return

        # If a monitoring schedule already exists, give an informative message
        if self.monitoring_schedule_exists():
            self.log.warning(f"Monitoring schedule for {self.endpoint_name} already exists.")
            self.log.warning("If you want to create another one, delete existing schedule first.")
            return

        # Set up a NEW monitoring schedule
        schedule_args = {
            "monitor_schedule_name": self.monitoring_schedule_name,
            "endpoint_input": self.endpoint_name,
            "output_s3_uri": self.monitoring_path,
            "statistics": self.statistics_json_file,
            "constraints": self.constraints_json_file,
            "schedule_cron_expression": schedule,
        }

        # Add preprocessing script to get rid of 'extra_column_check' violation (so stupid)
        feature_list = self.get_baseline().columns.to_list()
        script = preprocessing_script(feature_list)
        write_to_s3(script, self.preprocessing_script_file)
        self.log.important(f"Using preprocessing script: {self.preprocessing_script_file}")
        schedule_args["record_preprocessor_script"] = self.preprocessing_script_file

        # Create the monitoring schedule
        self.model_monitor.create_monitoring_schedule(**schedule_args)
        self.log.important(f"New Monitoring schedule created for {self.endpoint_name}.")

    def monitoring_schedule_exists(self):
        """Check if a monitoring schedule already exists for this endpoint

        Returns:
            bool: True if a monitoring schedule exists, False otherwise
        """
        try:
            self.sagemaker_client.describe_monitoring_schedule(MonitoringScheduleName=self.monitoring_schedule_name)
            return True
        except self.sagemaker_client.exceptions.ResourceNotFound:
            self.log.info(f"No monitoring schedule exists for {self.endpoint_name}.")
            return False

    def delete_monitoring_schedule(self):
        """Delete the monitoring schedule associated with this endpoint"""
        if not self.monitoring_schedule_exists():
            self.log.warning(f"No monitoring schedule exists for {self.endpoint_name}.")
            return

        # Use the model_monitor to delete the schedule
        self.model_monitor.delete_monitoring_schedule()
        self.log.important(f"Deleted monitoring schedule for {self.endpoint_name}.")

    # Put this functionality into this class
    """
    executions = my_monitor.list_executions()
    latest_execution = executions[-1]

    latest_execution.describe()['ProcessingJobStatus']
    latest_execution.describe()['ExitMessage']
    Here are the possible terminal states and what each of them means:

    - Completed - This means the monitoring execution completed and no issues were found in the violations report.
    - CompletedWithViolations - This means the execution completed, but constraint violations were detected.
    - Failed - The monitoring execution failed, maybe due to client error
                (perhaps incorrect role premissions) or infrastructure issues. Further
                examination of the FailureReason and ExitMessage is necessary to identify what exactly happened.
    - Stopped - job exceeded the max runtime or was manually stopped.
    You can also get the S3 URI for the output with latest_execution.output.destination and analyze the results.

    Visualize results
    You can use the monitor object to gather reports for visualization:

    suggested_constraints = my_monitor.suggested_constraints()
    baseline_statistics = my_monitor.baseline_statistics()

    latest_monitoring_violations = my_monitor.latest_monitoring_constraint_violations()
    latest_monitoring_statistics = my_monitor.latest_monitoring_statistics()
    """

    def get_monitoring_results(self, max_results=10) -> pd.DataFrame:
        """Get the results of monitoring executions

        Args:
            max_results (int): Maximum number of results to return

        Returns:
            pd.DataFrame: DataFrame of monitoring results (empty if no schedule exists)
        """
        if not self.monitoring_schedule_exists():
            self.log.warning(f"No monitoring schedule exists for {self.endpoint_name}.")
            return pd.DataFrame()

        try:
            # Get the monitoring executions
            executions = self.sagemaker_client.list_monitoring_executions(
                MonitoringScheduleName=self.monitoring_schedule_name,
                MaxResults=max_results,
                SortBy="ScheduledTime",
                SortOrder="Descending",
            )

            # Extract the execution details
            execution_details = []
            for execution in executions.get("MonitoringExecutionSummaries", []):
                execution_arn = execution.get("MonitoringExecutionArn")
                if execution_arn:
                    detail = {
                        "execution_arn": execution_arn,
                        "status": execution.get("MonitoringExecutionStatus"),
                        "scheduled_time": execution.get("ScheduledTime"),
                        "creation_time": execution.get("CreationTime"),
                        "last_modified_time": execution.get("LastModifiedTime"),
                        "failure_reason": execution.get("FailureReason"),
                        "monitoring_type": execution.get("MonitoringType"),
                    }

                    # For failure cases, get additional details
                    if detail["status"] == "Failed":
                        try:
                            exec_details = self.sagemaker_client.describe_monitoring_execution(
                                MonitoringExecutionArn=execution_arn
                            )
                            detail.update(
                                {
                                    "failure_details": exec_details.get("FailureReason", ""),
                                    "processing_job_arn": exec_details.get("ProcessingJobArn"),
                                }
                            )
                        except Exception as e:
                            self.log.error(f"Error getting failure details: {e}")

                    # Get violation details if available
                    if detail["status"] == "Completed":
                        try:
                            # Check for violations
                            result_path = f"{self.monitoring_path}/{detail['creation_time'].strftime('%Y/%m/%d')}"
                            result_path += "/constraint_violations.json"
                            if wr.s3.does_object_exist(result_path):
                                violations_json = read_from_s3(result_path)
                                violations = parse_monitoring_results(violations_json)
                                detail["violations"] = violations.get("constraint_violations", [])
                                detail["violation_count"] = len(detail["violations"])
                            else:
                                detail["violations"] = []
                                detail["violation_count"] = 0
                        except Exception as e:
                            self.log.warning(f"Error getting violations: {e}")
                            detail["violations"] = []
                            detail["violation_count"] = -1

                    execution_details.append(detail)

            return pd.DataFrame(execution_details)

        except Exception as e:
            self.log.error(f"Error getting monitoring results: {e}")
            return pd.DataFrame()

    def setup_alerts(self, notification_email, threshold=1):
        """Set up CloudWatch alarms for monitoring violations with email notifications

        Args:
            notification_email (str): Email to send notifications
            threshold (int): Number of violations to trigger an alarm

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.monitoring_schedule_exists():
            self.log.warning(f"No monitoring schedule exists for {self.endpoint_name}.")
            return False

        try:
            # Create CloudWatch client
            boto3_session = self.sagemaker_session.boto_session
            cloudwatch_client = boto3_session.client("cloudwatch")
            sns_client = boto3_session.client("sns")

            # Create a complete alarm configuration with required parameters
            alarm_name = f"{self.endpoint_name}-monitoring-violations"

            # Create CloudWatch alarm configuration
            alarm_config = {
                "AlarmName": alarm_name,
                "AlarmDescription": f"Monitoring violations for {self.endpoint_name}",
                "MetricName": "ModelDataQualityMonitorViolations",
                "Namespace": "AWS/SageMaker",
                "Statistic": "Maximum",
                "Dimensions": [
                    {"Name": "MonitoringSchedule", "Value": self.monitoring_schedule_name},
                    {"Name": "EndpointName", "Value": self.endpoint_name},
                ],
                "Period": 300,  # 5 minutes
                "EvaluationPeriods": 1,
                "Threshold": threshold,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            }

            # Create SNS topic for notifications
            topic_name = f"{self.endpoint_name}-monitoring-alerts"
            topic_response = sns_client.create_topic(Name=topic_name)
            topic_arn = topic_response["TopicArn"]

            # Subscribe email to topic
            sns_client.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=notification_email)

            # Add SNS topic to alarm actions
            alarm_config["AlarmActions"] = [topic_arn]

            # Create the alarm with the complete configuration
            cloudwatch_client.put_metric_alarm(**alarm_config)

            self.log.important(f"Set up CloudWatch alarm with email notification to {notification_email}")
            return True
        except Exception as e:
            self.log.error(f"Error setting up CloudWatch alarm: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of this MonitorCore object

        Returns:
            str: String representation of this MonitorCore object
        """
        summary_dict = {}  # Disabling for now self.summary()
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
    mm = MonitorCore(endpoint_name)

    # Check the summary of the monitoring class
    pprint(mm.summary())

    # Check the details of the monitoring class
    pprint(mm.details())

    # Enable data capture on the endpoint
    mm.enable_data_capture()

    # Create a baseline for monitoring
    # mm.create_baseline(recreate=True)
    mm.create_baseline()

    # Check the monitoring outputs
    print("\nBaseline outputs...")
    base_df = mm.get_baseline()
    print(base_df.head())

    print("\nConstraints...")
    pprint(mm.get_constraints())

    print("\nStatistics...")
    print(mm.get_statistics())

    # Set up the monitoring schedule (if it doesn't already exist)
    mm.create_monitoring_schedule()

    #
    # Test the data capture by running some predictions
    #

    # Make predictions on the Endpoint using the FeatureSet evaluation data
    pred_df = my_endpoint.auto_inference()
    print(pred_df.head())

    # Check that data capture is working
    input_df, output_df = mm.get_captured_data()
    if input_df.empty or output_df.empty:
        print("No data capture files found, for a new endpoint it may take a few minutes to start capturing data")
    else:
        print("Found data capture files")
        print("Input")
        print(input_df.head())
        print("Output")
        print(output_df.head())

    # Test update_constraints
    print("\nTesting constraint updates...")
    custom_constraints = {"sex": {"allowed_values": ["M", "F", "I"]}, "length": {"min": 0.0, "max": 1.0}}
    mm.update_constraints(custom_constraints)

    # Test monitoring results retrieval
    print("\nTesting monitoring results retrieval...")
    results_df = mm.get_monitoring_results(max_results=5)
    if not results_df.empty:
        print(f"Found {len(results_df)} monitoring executions")
        print(results_df[["status", "scheduled_time", "violation_count"]].head())
    else:
        print("No monitoring results found yet")

    # Test alert setup
    print("\nTesting alert setup...")
    # mm.setup_alerts(notification_email="support@supercowpowers.com", threshold=2)

    # Test deleting and recreate the monitoring schedule
    # Note: These are commented out for now

    print("\nDeleting monitoring schedule...")
    # mm.delete_monitoring_schedule()

    print("Recreating the monitoring schedule.")
    # mm.create_monitoring_schedule()
