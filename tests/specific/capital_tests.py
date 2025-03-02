import pytest
from workbench.api.data_source import DataSource
from pprint import pprint


@pytest.mark.long
def test():
    # Create a new Data Source from an S3 Path (or a local file)
    source_path = "s3://workbench-public-data/common/abalone.csv"
    my_data = DataSource(source_path)
    pprint(my_data.summary())
    pprint(my_data.details())

    # Create a FeatureSet (with a name that has mixed case)
    fs = my_data.to_features("aBaLone-feaTures")
    assert fs is None


if __name__ == "__main__":
    test()
