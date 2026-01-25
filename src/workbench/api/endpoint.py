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

    def inference(
        self,
        eval_df: pd.DataFrame,
        capture_name: str = None,
        id_column: str = None,
        drop_error_rows: bool = False,
        include_quantiles: bool = False,
    ) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            capture_name (str, optional): The Name of the capture to use (default: None)
            id_column (str, optional): The name of the column to use as the ID (default: None)
            drop_error_rows (bool): Whether to drop rows with errors (default: False)
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().inference(eval_df, capture_name, id_column, drop_error_rows, include_quantiles)

    def auto_inference(self) -> pd.DataFrame:
        """Run inference on the Endpoint using the test data from the model training view

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().auto_inference()

    def full_inference(self) -> pd.DataFrame:
        """Run inference on the Endpoint using the full data from the model training view

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().full_inference()

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

    def cross_fold_inference(self, include_quantiles: bool = False) -> pd.DataFrame:
        """Pull cross-fold inference from model associated with this Endpoint

        Args:
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: A DataFrame with cross fold predictions
        """
        return super().cross_fold_inference(include_quantiles)


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
