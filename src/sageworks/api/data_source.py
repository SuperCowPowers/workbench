"""DataSource: Manages AWS Data Catalog creation and management.
DataSources are set up so that can easily be queried with AWS Athena.
All DataSources are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) DataSources
can be viewed and explored within the SageWorks Dashboard UI."""

import os
import pandas as pd
import logging

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.core.artifacts.athena_source import AthenaSource
from sageworks.core.transforms.data_loaders.light.csv_to_data_source import CSVToDataSource
from sageworks.core.transforms.data_loaders.light.s3_to_data_source_light import S3ToDataSourceLight
from sageworks.core.transforms.pandas_transforms.pandas_to_data import PandasToData
from sageworks.core.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.api.feature_set import FeatureSet
from sageworks.utils.aws_utils import extract_data_source_basename


class DataSource(AthenaSource):
    """DataSource: SageWorks DataSource API Class

    Common Usage:
        ```
        my_data = DataSource(name_of_source)
        my_data.details()
        my_features = my_data.to_features()
        ```
    """

    def __init__(self, source, name: str = None, tags: list = None):
        """
        Initializes a new DataSource object.

        Args:
            source (str): The source of the data. This can be an S3 bucket, file path,
                          DataFrame object, or an existing DataSource object.
            name (str): The name of the data source (must be lowercase). If not specified, a name will be generated
            tags (list[str]): A list of tags associated with the data source. If not specified tags will be generated.
        """
        self.log = logging.getLogger("sageworks")

        # Automatically generate a name if not provided
        ds_name = extract_data_source_basename(source) if name is None else name
        ds_name = Artifact.base_compliant_uuid(ds_name)  # Make sure UUID is compliant

        # Make sure we have a name for when we use a DataFrame source
        if ds_name == "dataframe":
            msg = "Set the 'name' argument in the constructor: DataSource(df, name='my_data')"
            self.log.critical(msg)
            raise ValueError(msg)

        # Set the tags and load the source
        tags = [ds_name] if tags is None else tags
        self._load_source(source, ds_name, tags)

        # Call superclass init
        super().__init__(ds_name)

    def details(self, **kwargs) -> dict:
        """DataSource Details

        Returns:
            dict: A dictionary of details about the DataSource
        """
        return super().details(**kwargs)

    def query(self, query: str) -> pd.DataFrame:
        """Query the AthenaSource

        Args:
            query (str): The query to run against the DataSource

        Returns:
            pd.DataFrame: The results of the query
        """
        return super().query(query)

    def pull_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this DataSource

        Returns:
            pd.DataFrame: A DataFrame of ALL the data from this DataSource

        Note:
            Obviously, this is not recommended for large datasets :)
        """

        # Get the table associated with the data
        self.log.info(f"Pulling all data from {self.uuid}...")
        table = super().get_table_name()
        query = f"SELECT * FROM {table}"
        return self.query(query)

    def to_features(
        self,
        name: str = None,
        tags: list = None,
        target_column: str = None,
        id_column: str = None,
        event_time_column: str = None,
        auto_one_hot: bool = False,
    ) -> FeatureSet:
        """
        Convert the DataSource to a FeatureSet

        Args:
            name (str): Set the name for feature set (must be lowercase). If not specified, a name will be generated
            tags (list): Set the tags for the feature set. If not specified tags will be generated.
            target_column (str): Set the target column for the feature set. (Optional)
            id_column (str): Set the id column for the feature set. If not specified will be generated.
            event_time_column (str): Set the event time for the feature set. If not specified will be generated.
            auto_one_hot (bool): Automatically one-hot encode categorical fields (default: False)

        Returns:
            FeatureSet: The FeatureSet created from the DataSource
        """

        # Create the FeatureSet Name
        fs_name = self.uuid.replace("_data", "") + "_features" if name is None else name
        fs_name = Artifact.base_compliant_uuid(fs_name)  # Make sure UUID is compliant

        # Set the Tags
        tags = [fs_name] if tags is None else tags

        # Transform the DataSource to a FeatureSet
        data_to_features = DataToFeaturesLight(self.uuid, fs_name)
        data_to_features.set_output_tags(tags)
        data_to_features.transform(
            target_column=target_column,
            id_column=id_column,
            event_time_column=event_time_column,
            auto_one_hot=auto_one_hot,
        )

        # Return the FeatureSet (which will now be up-to-date)
        return FeatureSet(fs_name)

    def _load_source(self, source: str, name: str, tags: list):
        """Load the source of the data"""
        self.log.info(f"Loading source: {source}...")

        # Pandas DataFrame Source
        if isinstance(source, pd.DataFrame):
            my_loader = PandasToData(name)
            my_loader.set_input(source)
            my_loader.set_output_tags(tags)
            my_loader.transform()

        # S3 Source
        source = source if isinstance(source, str) else str(source)
        if source.startswith("s3://"):
            my_loader = S3ToDataSourceLight(source, name)
            my_loader.set_output_tags(tags)
            my_loader.transform()

        # File Source
        elif os.path.isfile(source):
            my_loader = CSVToDataSource(source, name)
            my_loader.set_output_tags(tags)
            my_loader.transform()


if __name__ == "__main__":
    """Exercise the DataSource Class"""
    import sys
    from pathlib import Path
    from pprint import pprint
    from sageworks.utils.test_data_generator import TestDataGenerator

    # Test to Run
    long_tests = False

    # Retrieve an existing Data Source
    test_data = DataSource("test_data")
    if test_data.exists():
        pprint(test_data.summary())
        pprint(test_data.details())

    # Create a new Data Source from a Pandas DataFrame
    gen_data = TestDataGenerator()
    df = gen_data.person_data()
    test_data = DataSource(df, name="test_data")
    pprint(test_data.summary())

    # Example of using the Parent Class/Core API for lower level access
    my_data = DataSource("abalone_data")
    pprint(my_data.outliers().head())

    # Long Tests
    if long_tests:
        # Create a new Data Source from a CSV file
        abalone_data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "abalone.csv"
        my_data = DataSource(abalone_data_path)
        pprint(my_data.summary())
        pprint(my_data.details())

        # Create a new Data Source from an S3 Path
        my_data = DataSource("s3://sageworks-public-data/common/abalone.csv")
        pprint(my_data.summary())
        pprint(my_data.details())

        # Convert the Data Source to a Feature Set
        my_features = test_data.to_features()
        pprint(my_features.summary())
        pprint(my_features.details())
