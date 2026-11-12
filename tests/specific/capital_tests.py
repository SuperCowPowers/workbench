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

    # A FeatureSet name with mixed case and dashes is rejected
    with pytest.raises(ValueError, match="Invalid FeatureSet name"):
        my_data.to_features("aBaLone-feaTures-Temp", id_column="auto")


if __name__ == "__main__":
    test()
