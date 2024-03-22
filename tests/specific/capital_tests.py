import pytest
from sageworks.api.data_source import DataSource
from pprint import pprint


@pytest.mark.long
def test():
    # Create a new Data Source from an S3 Path (or a local file)
    source_path = "s3://sageworks-public-data/common/aBaLone.CSV"
    # source_path = "/full/path/to/local/file.csv"
    my_data = DataSource(source_path)
    pprint(my_data.summary())
    pprint(my_data.details())

    # Create a FeatureSet (with a name that has mixed case)
    try:
        my_data.to_features("aBaLone-feaTures")
        assert False  # Should not get here
    except ValueError:
        pass


if __name__ == "__main__":
    test()
