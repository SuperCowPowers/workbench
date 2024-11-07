"""DataSource: Manages AWS Data Catalog creation and management.
DataSources are set up so that can easily be queried with AWS Athena.
All DataSources are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) DataSources
can be viewed and explored within the SageWorks Dashboard UI."""

import os
import pandas as pd
from typing import Union

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
        ```python
        my_data = DataSource(name_of_source)
        my_data.details()
        my_features = my_data.to_features()
        ```
    """

    def __init__(self, source: Union[str, pd.DataFrame], name: str = None, tags: list = None, **kwargs):
        """
        Initializes a new DataSource object.

        Args:
            source (Union[str, pd.DataFrame]): Source of data (existing name, filepath, S3 path, or a Pandas DataFrame)
            name (str): The name of the data source (must be lowercase). If not specified, a name will be generated
            tags (list[str]): A list of tags associated with the data source. If not specified tags will be generated.
        """

        # Ensure the ds_name is valid
        if name:
            Artifact.is_name_valid(name)

        # If the data source name wasn't given, generate it
        else:
            name = extract_data_source_basename(source)
            name = Artifact.generate_valid_name(name)

            # Sanity check for dataframe sources
            if name == "dataframe":
                msg = "Set the 'name' argument in the constructor: DataSource(df, name='my_data')"
                self.log.critical(msg)
                raise ValueError(msg)

        # Set the tags and load the source
        tags = [name] if tags is None else tags
        self._load_source(source, name, tags)

        # Call superclass init
        super().__init__(name, **kwargs)

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

    def pull_dataframe(self, include_aws_columns=False) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this DataSource

        Args:
            include_aws_columns (bool): Include the AWS columns in the DataFrame (default: False)

        Returns:
            pd.DataFrame: A DataFrame of ALL the data from this DataSource

        Note:
            Obviously this is not recommended for large datasets :)
        """

        # Get the table associated with the data
        self.log.info(f"Pulling all data from {self.uuid}...")
        table = super().table
        query = f'SELECT * FROM "{table}"'
        df = self.query(query)

        # Drop any columns generated from AWS
        if not include_aws_columns:
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
            df = df.drop(columns=aws_cols, errors="ignore")
        return df

    def to_features(
        self,
        name: str,
        id_column: str,
        tags: list = None,
        event_time_column: str = None,
        one_hot_columns: list = None,
    ) -> Union[FeatureSet, None]:
        """
        Convert the DataSource to a FeatureSet

        Args:
            name (str): Set the name for feature set (must be lowercase).
            id_column (str): The ID column (must be specified, use "auto" for auto-generated IDs).
            tags (list, optional: Set the tags for the feature set. If not specified tags will be generated
            event_time_column (str, optional): Set the event time for feature set. If not specified will be generated
            one_hot_columns (list, optional): Set the columns to be one-hot encoded. (default: None)

        Returns:
            FeatureSet: The FeatureSet created from the DataSource (or None if the FeatureSet isn't created)
        """

        # Ensure the feature_set_name is valid
        if not Artifact.is_name_valid(name):
            self.log.critical(f"Invalid FeatureSet name: {name}, not creating FeatureSet!")
            return None

        # Set the Tags
        tags = [name] if tags is None else tags

        # Transform the DataSource to a FeatureSet
        data_to_features = DataToFeaturesLight(self.uuid, name)
        data_to_features.set_output_tags(tags)
        data_to_features.transform(
            id_column=id_column,
            event_time_column=event_time_column,
            one_hot_columns=one_hot_columns,
        )

        # Return the FeatureSet (which will now be up-to-date)
        return FeatureSet(name)

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

    # Check logic for creating a new DataSource with DataFrame (without name)
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    try:
        DataSource(df)
    except ValueError:
        print("AOK!")

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
