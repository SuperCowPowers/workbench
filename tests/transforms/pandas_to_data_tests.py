"""Tests for the Pandas DataFrame to Data Transforms"""

# Workbench imports
from workbench.core.transforms.pandas_transforms import PandasToData
from workbench.utils.test_data_generator import TestDataGenerator


def test():
    """Tests for the Pandas DataFrame to Data Transforms"""

    # Generate some test data
    test_data = TestDataGenerator()
    df = test_data.person_data()

    # Create my Pandas to DataSource Transform
    test_name = "test_data"
    df_to_data = PandasToData(test_name)
    df_to_data.set_input(df)
    df_to_data.set_output_tags(["test", "small"])
    df_to_data.transform()
    print(f"{test_name} stored as a Workbench DataSource")


if __name__ == "__main__":
    test()
