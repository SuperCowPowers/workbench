"""Tests for the Data to Data (light) Transforms"""

import pytest

# Local imports
from workbench.core.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from workbench.api.data_source import DataSource


# Simple test of the DataToFeaturesLight functionality
@pytest.mark.long
def transform_test():
    """Tests for the Data to Features (light) Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_features"
    data_to_features = DataToFeaturesLight(input_uuid, output_uuid)
    data_to_features.set_output_tags(["test", "small"])
    data_to_features.transform(id_column="id", event_time_column="date")


# Testing the DataSource API to_features() method
@pytest.mark.long
def to_features_test():

    ds = DataSource("wine_data")
    ds.to_features("wine_features", tags=["wine", "classification"])


if __name__ == "__main__":
    transform_test()
    to_features_test()
