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
    test_uuid = "test_features"
    df_to_data = PandasToFeatures(test_uuid)
    df_to_data.set_input(df, id_column="id")
    df_to_data.set_output_tags(["test", "small"])
    df_to_data.transform()
    print(f"{test_uuid} stored as a Workbench FeatureSet")

    # Set holdout ids
    fs = FeatureSet(test_uuid)
    fs.set_training_holdouts("id", [1, 2, 3])


if __name__ == "__main__":
    test()
