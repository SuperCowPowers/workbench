"""Endpoint: Manages AWS Endpoint creation and deployment.
Endpoints are automatically set up and provisioned for deployment into AWS.
Endpoints can be viewed in the AWS Sagemaker interfaces or in the SageWorks
Dashboard UI, which provides additional model details and performance metrics"""

import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.endpoint_core import EndpointCore


class Endpoint(EndpointCore):
    """Endpoint: SageWorks Endpoint API Class

    Common Usage:
        ```
        my_endpoint = Endpoint(name)
        my_endpoint.details()
        my_endpoint.predict(df)
        ```
    """

    def details(self, **kwargs) -> dict:
        """Endpoint Details

        Returns:
            dict: A dictionary of details about the Endpoint
        """
        return super().details(**kwargs)

    def predict(self, df) -> pd.DataFrame:
        """Run predictions on the Endpoint

        Args:
            df (pd.DataFrame): The DataFrame to run predictions on

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().predict(df)


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from sageworks.api.model import Model
    from sageworks.api.feature_set import FeatureSet
    from pprint import pprint

    # Retrieve an existing Data Source
    my_endpoint = Endpoint("test-end")
    pprint(my_endpoint.summary())

    # Run predictions on the Endpoint
    model_name = my_endpoint.get_input()
    fs_name = Model(model_name).get_input()
    my_features = FeatureSet(fs_name)
    table = my_features.get_training_view_table()
    df = my_features.query(f"SELECT * FROM {table} where training = 0")
    results = my_endpoint.predict(df)
    pprint(results[["iq_score", "prediction"]])
