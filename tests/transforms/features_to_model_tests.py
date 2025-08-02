"""Tests for the Features to Model Transforms"""

import pytest

# Local Imports
from workbench.api import FeatureSet
from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
from workbench.core.artifacts.model_core import ModelType


# Simple test of the FeaturesToModel functionality
@pytest.mark.long
def test():
    """Test the Features to Model Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_name = "abalone_features"
    output_name = "abalone-regression-temp"
    to_model = FeaturesToModel(input_name, output_name, ModelType.REGRESSOR)
    to_model.set_output_tags(["temp", "abalone", "public"])
    to_model.transform(target_column="class_number_of_rings", description="Abalone Regression")


@pytest.mark.long
def test_categorical():
    """Test the Features to Model Transforms"""

    # Create the Test Model (with categorical features)
    features = ["height", "weight", "salary", "age", "iq_score", "likes_dogs", "food"]  # Food is categorical
    fs = FeatureSet("test_features")
    m = fs.to_model(
        name="test-regression-temp",
        model_type=ModelType.REGRESSOR,
        feature_list=features,
        target_column="iq_score",
        tags=["temp", "test", "regression"],
        description="Test Model with Categorical Features",
    )
    m.set_owner("test")


if __name__ == "__main__":
    test()
