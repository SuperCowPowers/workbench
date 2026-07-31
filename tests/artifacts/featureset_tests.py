"""Tests for the FeatureSet functionality"""

# Workbench Imports
from workbench.core.artifacts.feature_set_core import FeatureSetCore


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
    # print('Deleting Workbench Feature Set...')
    # my_features.delete()


def test_query_table_overwrite():
    """query() resolves the FeatureSet name to the versioned Athena table"""
    my_features = FeatureSetCore("test_features")
    name, table = my_features.name, my_features.athena_table

    # Capture the rewritten SQL instead of running it against Athena
    captured = []
    my_features.data_source.query = lambda q: captured.append(q)

    def rewrite(query: str) -> str:
        captured.clear()
        my_features.query(query)
        return captured[0]

    # The name resolves wherever it sits: mid-line, end of string, before a newline, or quoted
    assert rewrite(f"SELECT * FROM {name} WHERE x = 1") == f"SELECT * FROM {table} WHERE x = 1"
    assert rewrite(f"SELECT *\nFROM {name}") == f"SELECT *\nFROM {table}"
    assert rewrite(f"SELECT *\nFROM {name}\nLIMIT 3") == f"SELECT *\nFROM {table}\nLIMIT 3"
    assert rewrite(f'SELECT * FROM "{name}"') == f'SELECT * FROM "{table}"'

    # Whole-word only: an already-resolved table and a longer sibling name are both left alone
    assert rewrite(f"SELECT * FROM {table}") == f"SELECT * FROM {table}"
    assert rewrite(f"SELECT * FROM {name}_holdout") == f"SELECT * FROM {name}_holdout"

    # overwrite=False opts out entirely
    captured.clear()
    my_features.query(f"SELECT * FROM {name}", overwrite=False)
    assert captured[0] == f"SELECT * FROM {name}"


if __name__ == "__main__":
    test()
    test_query_table_overwrite()
