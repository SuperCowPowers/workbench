"""Fast Inference on SageMaker Endpoints"""

import pandas as pd
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor

# Sagemaker Imports
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

log = logging.getLogger("workbench")

_CACHED_SM_SESSION = None


def get_or_create_sm_session():
    global _CACHED_SM_SESSION
    if _CACHED_SM_SESSION is None:
        _CACHED_SM_SESSION = sagemaker.Session()
    return _CACHED_SM_SESSION


def fast_inference(endpoint_name: str, eval_df: pd.DataFrame, sm_session=None, threads: int = 4) -> pd.DataFrame:
    """Run inference on the Endpoint using the provided DataFrame

    Args:
        endpoint_name (str): The name of the Endpoint
        eval_df (pd.DataFrame): The DataFrame to run predictions on
        sm_session (sagemaker.session.Session, optional): SageMaker Session. If None, a cached session is created.
        threads (int): The number of threads to use (default: 4)

    Returns:
        pd.DataFrame: The DataFrame with predictions
    """
    # Use cached session if none is provided
    if sm_session is None:
        sm_session = get_or_create_sm_session()

    predictor = Predictor(
        endpoint_name,
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )

    total_rows = len(eval_df)

    def process_chunk(chunk_df: pd.DataFrame, start_index: int) -> pd.DataFrame:
        log.info(f"Processing {start_index}:{min(start_index + chunk_size, total_rows)} out of {total_rows} rows...")
        csv_buffer = StringIO()
        chunk_df.to_csv(csv_buffer, index=False)
        response = predictor.predict(csv_buffer.getvalue())
        # CSVDeserializer returns a nested list: first row is headers
        return pd.DataFrame.from_records(response[1:], columns=response[0])

    # Sagemaker has a connection pool limit of 10
    if threads > 10:
        log.warning("Sagemaker has a connection pool limit of 10. Reducing threads to 10.")
        threads = 10

    # Compute the chunk size (divide number of threads)
    chunk_size = max(1, total_rows // threads)

    # We also need to ensure that the chunk size is not too big
    if chunk_size > 100:
        chunk_size = 100

    # Split DataFrame into chunks and process them concurrently
    chunks = [(eval_df[i : i + chunk_size], i) for i in range(0, total_rows, chunk_size)]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        df_list = list(executor.map(lambda p: process_chunk(*p), chunks))

    combined_df = pd.concat(df_list, ignore_index=True)

    # Convert the types of the dataframe
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
            pass

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
    from workbench.utils.endpoint_utils import fs_training_data, fs_evaluation_data

    # Create an Endpoint
    my_endpoint_name = "abalone-regression"
    my_endpoint = Endpoint(my_endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {my_endpoint_name} does not exist.")
        exit(1)

    # Get the training data
    my_train_df = fs_training_data(my_endpoint)
    print(my_train_df)

    # Run Fast Inference and time it
    my_sm_session = my_endpoint.sm_session
    my_eval_df = fs_evaluation_data(my_endpoint)
    start_time = time.time()
    my_results_df = fast_inference(my_endpoint_name, my_eval_df, my_sm_session)
    end_time = time.time()
    print(f"Fast Inference took {end_time - start_time} seconds")
    print(my_results_df)
    print(my_results_df.info())

    # Test with no session
    my_results_df = fast_inference(my_endpoint_name, my_eval_df)
    print(my_results_df)
    print(my_results_df.info())
