"""Tests for the creation and comparison of Model Metrics"""

import pytest
from pprint import pprint

# SageWorks Imports
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint


# Simple test of the FeaturesToModel functionality
@pytest.mark.long
def test():
    """Test the Features to Model Transforms"""

    # Retrieve an existing FeatureSet
    my_features = FeatureSet("test_features")
    pprint(my_features.summary())

    # Create a Model/Endpoint from the FeatureSet
    create_model = True
    if create_model:
        my_model = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="iq_score")
        my_endpoint = my_model.to_endpoint(name="test-end", tags=["test"])
    else:
        my_model = Model("test-model")
        my_endpoint = Endpoint("test-end")

    # Grab the model metrics
    metrics = my_model.model_metrics()
    pprint(metrics)

    # Run inference on the model with the FeatureSet
    pred_results = my_endpoint.auto_inference()
    pprint(pred_results)


if __name__ == "__main__":

    # Set Pandas display options
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    test()
