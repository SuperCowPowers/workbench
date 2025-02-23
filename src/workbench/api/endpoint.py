"""Endpoint: Manages AWS Endpoint creation and deployment.
Endpoints are automatically set up and provisioned for deployment into AWS.
Endpoints can be viewed in the AWS Sagemaker interfaces or in the Workbench
Dashboard UI, which provides additional model details and performance metrics"""

import pandas as pd

# Workbench Imports
from workbench.core.artifacts.endpoint_core import EndpointCore


class Endpoint(EndpointCore):
    """Endpoint: Workbench Endpoint API Class

    Common Usage:
        ```python
        my_endpoint = Endpoint(name)
        my_endpoint.details()
        my_endpoint.inference(eval_df)
        ```
    """

    def details(self, **kwargs) -> dict:
        """Endpoint Details

        Returns:
            dict: A dictionary of details about the Endpoint
        """
        return super().details(**kwargs)

    def inference(self, eval_df: pd.DataFrame, capture_uuid: str = None, id_column: str = None) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            capture_uuid (str, optional): The UUID of the capture to use (default: None)
            id_column (str, optional): The name of the column to use as the ID (default: None)

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().inference(eval_df, capture_uuid, id_column)

    def auto_inference(self, capture: bool = False) -> pd.DataFrame:
        """Run inference on the Endpoint using the FeatureSet evaluation data

        Args:
            capture (bool): Capture the inference results

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().auto_inference(capture)

    def fast_inference(self, eval_df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            threads (int): The number of threads to use (default: 4)

        Returns:
            pd.DataFrame: The DataFrame with predictions

        Note:
            There's no sanity checks or error handling... just FAST Inference!
        """
        return super().fast_inference(eval_df, threads=threads)


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from workbench.api.model import Model
    from workbench.api.feature_set import FeatureSet
    from pprint import pprint

    # Retrieve an existing Data Source
    my_endpoint = Endpoint("abalone-regression")
    pprint(my_endpoint.summary())

    # Run predictions on the Endpoint
    model = Model(my_endpoint.get_input())
    my_features = FeatureSet(model.get_input())
    table = my_features.view("training").table
    df = my_features.query(f'SELECT * FROM "{table}" where training = FALSE')
    results = my_endpoint.inference(df)
    target = model.target()
    pprint(results[[target, "prediction"]])

    # Run predictions using the auto_inference method
    auto_results = my_endpoint.auto_inference()
    pprint(auto_results[[target, "prediction"]])

    # Run predictions using the fast_inference method
    fast_results = my_endpoint.fast_inference(df)
