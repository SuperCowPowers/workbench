"""Tests for the Features to Model Transforms"""
import pytest

# Local Imports
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel


# Simple test of the FeaturesToModel functionality
@pytest.mark.slow
def test():
    """Tests for the Features to Model Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone_feature_set"
    output_uuid = "abalone-regression"
    to_model = FeaturesToModel(input_uuid, output_uuid)
    to_model.set_output_tags(["abalone", "public"])
    to_model.transform(target="class_number_of_rings")


if __name__ == "__main__":
    test()
