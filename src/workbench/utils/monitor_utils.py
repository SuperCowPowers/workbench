"""Utility functions for SageMaker model monitoring"""

import json
import logging
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, Any, Union
from io import StringIO
import awswrangler as wr

# Workbench Imports
from workbench.utils.s3_utils import read_content_from_s3

# Setup logging
log = logging.getLogger("workbench")

# SageMaker stores captured payloads base64-encoded unless the endpoint's capture config
# declares the exact content types. Responses carry a charset suffix, so both forms are needed.
CAPTURE_CSV_CONTENT_TYPES = ["text/csv", "text/csv; charset=utf-8"]
CAPTURE_JSON_CONTENT_TYPES = ["application/json", "application/json; charset=utf-8"]


def pull_data_capture_for_testing(data_capture_path, max_files=1) -> Union[pd.DataFrame, None]:
    """
    Read and process captured data from S3.

    Args:
        data_capture_path (str): S3 path to the data capture files.
        max_files (int, optional): Maximum number of files to process.
                                  Defaults to 1 (most recent only).
                                  Set to None to process all files.

    Returns:
        Union[pd.DataFrame, None]: A dataframe of the captured data (or None if no data is found).

    Notes:
        This method is really only for testing and debugging.
    """
    log.important("This method is for testing and debugging only.")

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


def extract_capture_payloads(df: pd.DataFrame) -> tuple[list, list]:
    """
    Extract the raw input and output payloads from captured data records.

    Args:
        df (DataFrame): DataFrame with captured data.
    Returns:
        tuple[list, list]: Input and output payload dicts.
    """
    input_payloads = []
    output_payloads = []

    # Use itertuples() instead of iterrows() for better performance
    for row in df.itertuples(index=True):
        try:
            capture_data = row.captureData
            if "endpointInput" in capture_data:
                input_payloads.append(capture_data["endpointInput"])
            if "endpointOutput" in capture_data:
                output_payloads.append(capture_data["endpointOutput"])
        except Exception as e:
            log.debug(f"Row {row.Index}: Failed to process row: {e}")
            continue

    return input_payloads, output_payloads


def parse_payloads(payloads: list) -> tuple[pd.DataFrame, Union[np.ndarray, None]]:
    """
    Parse capture payloads into a single DataFrame.

    Args:
        payloads (list): Capture payload dicts (endpointInput or endpointOutput entries).
    Returns:
        tuple[DataFrame, ndarray | None]: The parsed data, plus the originating payload index for
                                          each row (None when that mapping can't be established).
    """

    def parse_json(payload: dict) -> pd.DataFrame:
        """Parse a single JSON payload."""
        json_data = json.loads(payload["data"])
        if isinstance(json_data, dict):
            return pd.DataFrame({k: v if isinstance(v, list) else [v] for k, v in json_data.items()})
        return pd.DataFrame(json_data)

    frames = []
    row_sources = []  # Per-frame arrays of originating payload index
    sources_valid = True
    unsupported = Counter()

    # CSV payloads sharing a header are parsed in a single pass, so dtypes are inferred once
    # across every row and the result is block-consolidated instead of one block per column
    csv_groups = defaultdict(list)  # header -> [(payload index, body), ...]
    for index, payload in enumerate(payloads):
        encoding = payload["encoding"].upper()
        if encoding == "CSV":
            header, _, body = payload["data"].strip().partition("\n")
            csv_groups[header].append((index, body))
        elif encoding == "JSON":
            try:
                frames.append(parse_json(payload))
                row_sources.append(np.full(len(frames[-1]), index))
            except Exception as e:
                log.debug(f"Failed to parse JSON payload: {e}")
        else:
            unsupported[encoding] += 1

    # An unsupported encoding means the endpoint's capture content types are misconfigured
    for encoding, count in unsupported.items():
        log.warning(f"Skipped {count} capture payloads with unsupported encoding: {encoding}")

    for header, entries in csv_groups.items():
        bodies = [body for _, body in entries]
        try:
            frame = pd.read_csv(StringIO("\n".join([header, *bodies])), low_memory=False)
        except Exception as e:
            log.warning(f"Skipping {len(bodies)} payloads, CSV parse failed: {e}")
            continue

        frames.append(frame)

        # Each body contributes one row per line, which lets us map rows back to payloads.
        # Anything that breaks that (an embedded newline, a blank line) invalidates the mapping.
        counts = [body.count("\n") + 1 if body else 0 for body in bodies]
        if sum(counts) == len(frame):
            row_sources.append(np.repeat([index for index, _ in entries], counts))
        else:
            sources_valid = False

    if not frames:
        return pd.DataFrame(), np.array([], dtype=int)

    # Concatenating payloads with differing schemas aligns columns individually, leaving one
    # block per column; copy() consolidates so downstream operations aren't crippled
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True).copy()
    if not sources_valid:
        log.warning("Capture rows could not be mapped back to their payloads.")
        return combined, None
    return combined, np.concatenate(row_sources)


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
    raw_json = read_content_from_s3(s3_path)
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
    df = pull_data_capture_for_testing(mon.data_capture_path)
    print("Data Capture:")
    print(df.head())

    # Test processing data capture
    input_payloads, output_payloads = extract_capture_payloads(df)

    print("\nProcessed Input:")
    print(parse_payloads(input_payloads)[0])

    print("\nProcessed Output:")
    print(parse_payloads(output_payloads)[0])

    # Test preprocessing script
    script = preprocessing_script(["feature1", "feature2", "feature3"])
    print("\nPreprocessing Script:")
    print(script)
