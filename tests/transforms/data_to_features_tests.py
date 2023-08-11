"""Tests for the Data to Data (light) Transforms"""
import pytest

# Local imports
from sageworks.transforms.data_to_features.light.data_to_features_light import (
    DataToFeaturesLight,
)


# Simple test of the DataToFeaturesLight functionality
@pytest.mark.slow
def test():
    """Tests for the Data to Features (light) Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_feature_set"
    data_to_features = DataToFeaturesLight(input_uuid, output_uuid)
    data_to_features.set_output_tags(["test", "small"])
    data_to_features.transform(id_column="id", event_time_column="date")


if __name__ == "__main__":
    test()
