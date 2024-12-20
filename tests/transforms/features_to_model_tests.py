"""Tests for the Features to Model Transforms"""

import pytest

# Local Imports
from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
from workbench.core.artifacts.model_core import ModelType


# Simple test of the FeaturesToModel functionality
@pytest.mark.long
def test():
    """Test the Features to Model Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone_features"
    output_uuid = "abalone-regression"
    to_model = FeaturesToModel(input_uuid, output_uuid, ModelType.REGRESSOR)
    to_model.set_output_tags(["abalone", "public"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Regression")


if __name__ == "__main__":
    test()
