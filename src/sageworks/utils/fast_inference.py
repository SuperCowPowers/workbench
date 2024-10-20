"""Fast Inference on SageMaker Endpoints"""

import pandas as pd
from io import StringIO

# Sagemaker Imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor


def fast_inference(endpoint_name: str, eval_df: pd.DataFrame, sm_session) -> pd.DataFrame:
    """Run inference on the Endpoint using the provided DataFrame

    Args:
        endpoint_name (str): The name of the Endpoint
        eval_df (pd.DataFrame): The DataFrame to run predictions on
        sm_session (sagemaker.session.Session): The SageMaker Session

    Returns:
        pd.DataFrame: The DataFrame with predictions

    Note:
        There's no sanity checks or error handling... just FAST Inference!
    """
    predictor = Predictor(
        endpoint_name,
        sagemaker_session=sm_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )

    # Convert the DataFrame into a CSV buffer
    csv_buffer = StringIO()
    eval_df.to_csv(csv_buffer, index=False)

    # Send the CSV Buffer to the predictor
    results = predictor.predict(csv_buffer.getvalue())

    # Construct a DataFrame from the results
    results_df = pd.DataFrame.from_records(results[1:], columns=results[0])
    return results_df


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""
    from sageworks.api.endpoint import Endpoint
    from sageworks.utils.endpoint_utils import fs_training_data, fs_evaluation_data

    # Create an Endpoint
    my_endpoint_name = "abalone-regression-end"
    my_endpoint = Endpoint(my_endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {my_endpoint_name} does not exist.")
        exit(1)

    # Get the training data
    my_train_df = fs_training_data(my_endpoint)
    print(my_train_df)

    # Run Fast Inference
    sm_session = my_endpoint.sm_session
    my_eval_df = fs_evaluation_data(my_endpoint)
    my_results_df = fast_inference(my_endpoint_name, my_eval_df, sm_session)
    print(my_results_df)
