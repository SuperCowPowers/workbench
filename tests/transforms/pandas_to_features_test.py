"""Tests for the Pandas DataFrame to FeatureSet Transforms"""

import pytest

# Workbench imports
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.api import FeatureSet
from workbench.utils.test_data_generator import TestDataGenerator


@pytest.mark.long
def test():
    """Tests for the Pandas DataFrame to Data Transforms"""

    # Generate some test data
    test_data = TestDataGenerator()
    df = test_data.person_data()

    # Create my Pandas to DataSource Transform
    test_name = "test_features_temp"
    df_to_data = PandasToFeatures(test_name)
    df_to_data.set_input(df, id_column="id")
    df_to_data.set_output_tags(["temp", "test", "small"])
    df_to_data.transform()
    print(f"{test_name} stored as a Workbench FeatureSet")

    # Set holdout ids
    fs = FeatureSet(test_name)
    fs.set_training_holdouts([1, 2, 3])


if __name__ == "__main__":
    test()
