"""Fast Inference on SageMaker Endpoints"""

import sys
import os
import pandas as pd
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

log = logging.getLogger("workbench")

# SageMaker enforces a 60s server-side limit per invocation. read_timeout=90s
# gives headroom for network jitter on top of that so the client always receives
# the server's response (success or error) before giving up.
_SM_CLIENT_CONFIG = Config(connect_timeout=10, read_timeout=90)

_CACHED_SM_CLIENT = None
_CACHED_REGION = None


def get_aws_region():
    # Try environment variables first
    region = os.environ.get("SAGEMAKER_REGION") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    print(f"ENV REGION: {region}")
    region = region or boto3.session.Session().region_name
    if not region:
        msg = "No AWS region configured! Please set SAGEMAKER_REGION or AWS_REGION"
        log.critical(msg)
        raise Exception(msg)
    return region


def get_or_create_sm_client():
    """Get or create a cached SageMaker Runtime client."""
    global _CACHED_SM_CLIENT, _CACHED_REGION
    if _CACHED_SM_CLIENT is None:
        _CACHED_REGION = get_aws_region()
        print(f"Creating new SageMaker Runtime client in region: {_CACHED_REGION}")
        _CACHED_SM_CLIENT = boto3.Session(region_name=_CACHED_REGION).client(
            "sagemaker-runtime", config=_SM_CLIENT_CONFIG
        )
    return _CACHED_SM_CLIENT


def fast_inference(endpoint_name: str, eval_df: pd.DataFrame, sm_session=None, threads: int = 4) -> pd.DataFrame:
    """Run inference on the Endpoint using the provided DataFrame

    Args:
        endpoint_name (str): The name of the Endpoint
        eval_df (pd.DataFrame): The DataFrame to run predictions on
        sm_session: A boto3 Session (or legacy SageMaker Session with .boto_session). If None, a cached client is used.
        threads (int): The number of threads to use (default: 4)

    Returns:
        pd.DataFrame: The DataFrame with predictions
    """
    # Build the sagemaker-runtime client
    if sm_session is not None:
        # Support both plain boto3.Session and legacy SageMaker Session objects
        boto_session = getattr(sm_session, "boto_session", sm_session)
        sm_runtime = boto_session.client("sagemaker-runtime", config=_SM_CLIENT_CONFIG)
    else:
        sm_runtime = get_or_create_sm_client()

    total_rows = len(eval_df)

    # Fixed chunk size of 100 rows
    chunk_size = 100

    def process_chunk(chunk_df: pd.DataFrame, start_index: int) -> pd.DataFrame:
        log.info(f"Processing {start_index}:{min(start_index + chunk_size, total_rows)} out of {total_rows} rows...")
        csv_buffer = StringIO()
        chunk_df.to_csv(csv_buffer, index=False)
        try:
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=csv_buffer.getvalue(),
                ContentType="text/csv",
                Accept="text/csv",
            )
            # Read the response body and parse as CSV into a DataFrame
            response_body = response["Body"].read().decode("utf-8")
            # Handle 'charset' in content type (e.g. 'text/csv; charset=utf-8')
            return pd.read_csv(StringIO(response_body))
        except Exception as e:
            log.error(f"Error during prediction on '{endpoint_name}': {e}")
            return pd.DataFrame()

    # Sagemaker has a connection pool limit of 10
    if threads > 10:
        log.warning("Sagemaker has a connection pool limit of 10. Reducing threads to 10.")
        threads = 10

    # Split DataFrame into chunks and process them concurrently
    chunks = [(eval_df[i : i + chunk_size], i) for i in range(0, total_rows, chunk_size)]  # noqa: E203

    # Use min of threads or number of chunks to avoid creating unnecessary threads
    actual_threads = min(threads, len(chunks))
    with ThreadPoolExecutor(max_workers=actual_threads) as executor:
        df_list = list(executor.map(lambda p: process_chunk(*p), chunks))

    # Filter out empty DataFrames that might result from errors
    df_list = [df for df in df_list if not df.empty]

    if not df_list:
        raise RuntimeError(f"All prediction chunks failed for endpoint '{endpoint_name}'")

    combined_df = pd.concat(df_list, ignore_index=True)

    # Convert the column types of the dataframe (CSVDeserializer returns all columns as strings/objects)
    combined_df = df_type_conversions(combined_df)
    return combined_df


def df_type_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the types of the dataframe that we get from an endpoint

    Args:
        df (pd.DataFrame): DataFrame to convert

    Returns:
        pd.DataFrame: Converted DataFrame
    """
    # Some endpoints will put in "N/A" values (for CSV serialization)
    # We need to convert these to NaN and the run the conversions below
    # Report on the number of N/A values in each column in the DataFrame
    # For any count above 0 list the column name and the number of N/A values
    na_counts = df.isin(["N/A"]).sum()
    for column, count in na_counts.items():
        if count > 0:
            log.warning(f"{column} has {count} N/A values, converting to NaN")
    pd.set_option("future.no_silent_downcasting", True)
    df = df.replace("N/A", float("nan"))

    # Convert data to numeric
    # Note: Since we're using CSV serializers numeric columns often get changed to generic 'object' types

    # Hard Conversion
    # Note: We explicitly catch exceptions for columns that cannot be converted to numeric
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            # If a ValueError is raised, the column cannot be converted to numeric, so we keep it as is
            pass
        except TypeError:
            # This typically means a duplicated column name, so confirm duplicate (more than 1) and log it
            column_count = (df.columns == column).sum()
            log.critical(f"{column} occurs {column_count} times in the DataFrame.")

    # Soft Conversion
    # Convert columns to the best possible dtype that supports the pd.NA missing value.
    df = df.convert_dtypes()

    # Convert pd.NA placeholders to pd.NA
    # Note: CSV serialization converts pd.NA to blank strings, so we have to put in placeholders
    df.replace("__NA__", pd.NA, inplace=True)

    # Check for True/False values in the string columns
    for column in df.select_dtypes(include=["string"]).columns:
        if df[column].str.lower().isin(["true", "false"]).all():
            df[column] = df[column].str.lower().map({"true": True, "false": False})

    # Return the Dataframe
    return df


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""
    import time
    from workbench.api.endpoint import Endpoint
    from workbench.utils.endpoint_utils import get_evaluation_data

    # Grab the Endpoint
    my_endpoint_name = "abalone-regression"
    my_endpoint = Endpoint(my_endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {my_endpoint_name} does not exist.")
        sys.exit(1)

    # Pull evaluation data
    print("Pulling Evaluation Data...")
    sagemaker_session = my_endpoint.sm_session
    my_eval_df = get_evaluation_data(my_endpoint)
    start_time = time.time()
    my_results_df = fast_inference(my_endpoint_name, my_eval_df, sagemaker_session)
    end_time = time.time()
    print(f"Fast Inference took {end_time - start_time} seconds")
    print(my_results_df)
    print(my_results_df.info())

    # Test with no session
    my_results_df = fast_inference(my_endpoint_name, my_eval_df)
    print(my_results_df)
    print(my_results_df.info())
