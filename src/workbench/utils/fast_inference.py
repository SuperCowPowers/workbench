"""Fast Inference on SageMaker Endpoints"""

import pandas as pd
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor

# Sagemaker Imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

log = logging.getLogger("workbench")


def fast_inference(endpoint_name: str, eval_df: pd.DataFrame, sm_session, threads: int = 4) -> pd.DataFrame:
    """Run inference on the Endpoint using the provided DataFrame

    Args:
        endpoint_name (str): The name of the Endpoint
        eval_df (pd.DataFrame): The DataFrame to run predictions on
        sm_session (sagemaker.session.Session): The SageMaker Session
        threads (int): The number of threads to use (default: 4)

    Returns:
        pd.DataFrame: The DataFrame with predictions
    """
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

    return pd.concat(df_list, ignore_index=True)


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
    sm_session = my_endpoint.sm_session
    my_eval_df = fs_evaluation_data(my_endpoint)
    start_time = time.time()
    my_results_df = fast_inference(my_endpoint_name, my_eval_df, sm_session)
    end_time = time.time()
    print(f"Fast Inference took {end_time - start_time} seconds")
    print(my_results_df)
