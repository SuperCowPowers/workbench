"""Tests for the Pandas DataFrame to FeatureSet Transforms"""

import pytest

# Workbench imports
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.synthetic_data_generator import SyntheticDataGenerator


@pytest.mark.long
def test():
    """Tests for the Pandas DataFrame to Data Transforms"""

    # Generate some test data
    test_data = SyntheticDataGenerator()
    df = test_data.person_data()

    # Create my Pandas to DataSource Transform
    test_name = "pandas_features_test"
    df_to_data = PandasToFeatures(test_name)
    df_to_data.set_input(df, id_column="id")
    df_to_data.set_output_tags(["test", "small"])
    df_to_data.transform()
    print(f"{test_name} stored as a Workbench FeatureSet")


if __name__ == "__main__":
    test()
