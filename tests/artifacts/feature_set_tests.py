"""Tests for the FeatureSet functionality"""

# SageWorks Imports
from sageworks.core.artifacts.feature_set_core import FeatureSetCore


def test():
    """Simple test of the FeatureSet functionality"""
    from pprint import pprint

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSetCore("test_features")

    # Call the various methods

    # Let's do a check/validation of the feature set
    print(f"Feature Set Check: {my_features.exists()}")

    # How many rows and columns?
    num_rows = my_features.num_rows()
    num_columns = my_features.num_columns()
    print(f"Rows: {num_rows} Columns: {num_columns}")

    # What are the column names?
    print(my_features.columns)

    # Get Tags associated with this Feature Set
    print(f"Tags: {my_features.get_tags()}")

    # Get ALL the AWS Metadata associated with this Feature Set
    print("\n\nALL Meta")
    pprint(my_features.aws_meta())

    # Now delete the AWS artifacts associated with this Feature Set
    # print('Deleting SageWorks Feature Set...')
    # my_features.delete()


if __name__ == "__main__":
    test()
