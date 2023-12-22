"""DataSource: SageWorks DataSource API Class"""
import os
import logging

# SageWorks Imports
from sageworks.core.artifacts.athena_source import AthenaSource
from sageworks.core.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.core.transforms.data_loaders.light.s3_to_data_source_light import S3ToDataSourceLight
from sageworks.utils.aws_utils import extract_data_source_basename


class DataSource(AthenaSource):
    """DataSource: SageWorks DataSource API Class

    Common Usage:
        my_data = DataSource(name_of_source)
        my_data.summary()
        my_data.details()
        my_data.to_features()
    """

    def __init__(self, source, name=None):
        """DataSource Initialization
        Args:
            source (str): The source of the data (S3, File, or Existing DataSource)
            name (str): Set the name of the data source (optional)
        """
        self.log = logging.getLogger("sageworks")

        # Load the source (S3, File, or Existing DataSource)
        ds_name = extract_data_source_basename(source) if name is None else name
        self._load_source(source, ds_name)

        # Call superclass init
        super().__init__(ds_name)

    def _load_source(self, source: str, name: str):
        """Load the source of the data"""
        self.log.info(f"Loading source: {source}...")

        # Handle S3 path
        source = source if isinstance(source, str) else str(source)
        if source.startswith("s3://"):
            my_loader = S3ToDataSourceLight(source, name)
            my_loader.set_output_tags([name])
            my_loader.transform()
        elif os.path.isfile(source):
            my_loader = CSVToDataSource(source, name)
            my_loader.set_output_tags([name])
            my_loader.transform()


if __name__ == "__main__":
    """Exercise the DataSource Class"""
    import sys
    from pathlib import Path
    from pprint import pprint

    # Retrieve an existing Data Source
    my_data = DataSource("test_data")
    pprint(my_data.summary())
    pprint(my_data.details())

    # Create a new Data Source from a CSV file
    abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"
    my_data = DataSource(abalone_data_path)

    # Create a new Data Source from an S3 Path
    my_data = DataSource("s3://sageworks-public-data/common/abalone.csv")

    # Example of using the Parent Class/Core API for lower level access
    my_data = DataSource("abalone_data")
    print(my_data.outliers().head())
