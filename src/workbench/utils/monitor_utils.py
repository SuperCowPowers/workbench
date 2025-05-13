"""Utility functions for SageMaker model monitoring"""

import json
import logging
import pandas as pd
from typing import Dict, Any, Union
from io import StringIO
import awswrangler as wr

# Workbench Imports
from workbench.utils.s3_utils import read_from_s3

# Setup logging
log = logging.getLogger("workbench")


def pull_data_capture(data_capture_path, max_files=1) -> Union[pd.DataFrame, None]:
    """
    Read and process captured data from S3.

    Args:
        data_capture_path (str): S3 path to the data capture files.
        max_files (int, optional): Maximum number of files to process.
                                  Defaults to 1 (most recent only).
                                  Set to None to process all files.

    Returns:
        Union[pd.DataFrame, None]: A dataframe of the captured data (or None if no data is found).
    """
    # List files in the specified S3 path
    files = wr.s3.list_objects(data_capture_path)
    if not files:
        log.warning(f"No data capture files found in {data_capture_path}.")
        return None

    log.info(f"Found {len(files)} files in {data_capture_path}.")

    # Sort files by timestamp (assuming the naming convention includes timestamp)
    files.sort()

    # Select files to process
    if max_files is None:
        files_to_process = files
        log.info(f"Processing all {len(files)} files.")
    else:
        files_to_process = files[-max_files:] if files else []
        log.info(f"Processing the {len(files_to_process)} most recent file(s).")

    # Process each file
    all_data = []
    for file_path in files_to_process:
        try:
            # Read the JSON lines file
            df = wr.s3.read_json(path=file_path, lines=True)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            log.warning(f"Error processing file {file_path}: {e}")

    # Combine all DataFrames and return
    return pd.concat(all_data, ignore_index=True)


def process_data_capture(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the captured data DataFrame to extract input and output data.

    Args:
        df (DataFrame): DataFrame with captured data.

    Returns:
        tuple[DataFrame, DataFrame]: Input and output DataFrames.
    """
    input_dfs = []
    output_dfs = []

    for _, row in df.iterrows():
        capture_data = row["captureData"]

        # Extract and process input data
        input_data = capture_data["endpointInput"]
        if input_data["encoding"].upper() == "CSV":
            input_dfs.append(pd.read_csv(StringIO(input_data["data"])))
        elif input_data["encoding"].upper() == "JSON":
            json_data = json.loads(input_data["data"])
            if isinstance(json_data, dict):
                input_dfs.append(pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in json_data.items()}))
            else:
                input_dfs.append(pd.DataFrame(json_data))

        # Extract and process output data
        output_data = capture_data["endpointOutput"]
        if output_data["encoding"].upper() == "CSV":
            output_dfs.append(pd.read_csv(StringIO(output_data["data"])))
        elif output_data["encoding"].upper() == "JSON":
            json_data = json.loads(output_data["data"])
            if isinstance(json_data, dict):
                output_dfs.append(
                    pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in json_data.items()})
                )
            else:
                output_dfs.append(pd.DataFrame(json_data))

    # Combine and return results
    return (
        pd.concat(input_dfs, ignore_index=True) if input_dfs else pd.DataFrame(),
        pd.concat(output_dfs, ignore_index=True) if output_dfs else pd.DataFrame(),
    )


def get_monitor_json_data(s3_path: str) -> Union[dict, None]:
    """
    Convert JSON monitoring data into a DataFrame

    Args:
        s3_path (str): The S3 path to the monitoring data

    Returns:
        dict: A dictionary of the monitoring data (None if not found)
    """
    # Check if the S3 path exists
    if not wr.s3.does_object_exist(path=s3_path):
        log.warning(f"Monitoring data does not exist in S3: {s3_path}")
        return None

    # Read the JSON data from S3
    raw_json = read_from_s3(s3_path)
    return json.loads(raw_json)


def parse_monitoring_results(results_json: str) -> Dict[str, Any]:
    """
    Parse monitoring results from JSON

    Args:
        results_json (str): Monitoring results in JSON format

    Returns:
        dict: Parsed monitoring results
    """
    try:
        results = json.loads(results_json)

        # Extract and format the key information
        parsed_results = {
            "schema_validation": results.get("schema", {}).get("validation", {}),
            "constraint_violations": [],
        }

        # Extract violations
        for violation in results.get("violations", []):
            parsed_violation = {
                "feature_name": violation.get("feature_name"),
                "constraint_check_type": violation.get("constraint_check_type"),
                "description": violation.get("description"),
            }
            parsed_results["constraint_violations"].append(parsed_violation)

        return parsed_results
    except Exception as e:
        log.error(f"Error parsing monitoring results: {e}")
        return {"error": str(e)}


"""TEMP
                # If the status is "CompletedWithViolations", we grab the lastest
                # violation file and add it to the result
                if status == "CompletedWithViolations":
                    violation_file = f"{self.monitoring_path}/
                    {last_run['CreationTime'].strftime('%Y/%m/%d')}/constraint_violations.json"
                    if wr.s3.does_object_exist(violation_file):
                        violations_json = read_from_s3(violation_file)
                        violations = parse_monitoring_results(violations_json)
                        result["violations"] = violations.get("constraint_violations", [])
                        result["violation_count"] = len(result["violations"])
                    else:
                        result["violations"] = []
                        result["violation_count"] = 0
"""


def preprocessing_script(feature_list: list[str]) -> str:
    """
    A preprocessing script for monitoring jobs.

    Args:
        feature_list (list[str]): List of features to include in the preprocessing script.

    Returns:
        str: The preprocessing script
    """
    # Convert feature list to a proper Python string representation
    features_str = str(feature_list)

    script = """
import pandas as pd
from io import StringIO

def preprocess_handler(inference_record, logger):

    # CapturedData objects have endpoint_input with encoding and data attributes
    input_data = inference_record.endpoint_input.data

    # Parse the input data (assuming CSV format)
    df = pd.read_csv(StringIO(input_data))
    logger.info("Input DataFrame:")
    logger.info(df.shape)
    logger.info(df.columns)

    # Keep only the specified features
    # Note: That this feature list needs to be alphabetically sorted
    feature_list = {{features_str}}
    df = df[feature_list]

    logger.info("Output DataFrame:")
    logger.info(df.shape)
    logger.info(df.columns)
    output_data = df.to_dict(orient='records')
    return output_data
"""
    # Replace the placeholder with the actual feature list
    script = script.replace("{{features_str}}", features_str)
    return script


# Test function for the utils
if __name__ == "__main__":
    """Test the monitor_utils module"""
    from workbench.api.monitor import Monitor

    # Test pulling data capture
    mon = Monitor("abalone-regression-rt")
    df = pull_data_capture(mon.data_capture_path)
    print("Data Capture:")
    print(df.head())

    # Test processing data capture
    input_processed, output_processed = process_data_capture(df)

    print("\nProcessed Input:")
    print(input_processed)

    print("\nProcessed Output:")
    print(output_processed)

    # Test preprocessing script
    script = preprocessing_script(["feature1", "feature2", "feature3"])
    print("\nPreprocessing Script:")
    print(script)
