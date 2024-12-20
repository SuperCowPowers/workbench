"""PandasToFeaturesChunked: Class to manage a bunch of chunked Pandas DataFrames into a FeatureSet"""

import pandas as pd
from pandas.api.types import CategoricalDtype

# Local imports
from workbench.core.transforms.transform import Transform
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.artifacts.artifact import Artifact


class PandasToFeaturesChunked(Transform):
    """PandasToFeaturesChunked:  Class to manage a bunch of chunked Pandas DataFrames into a FeatureSet

    Common Usage:
        ```python
        to_features = PandasToFeaturesChunked(output_uuid, id_column="id"/None, event_time_column="date"/None)
        to_features.set_output_tags(["abalone", "public", "whatever"])
        cat_column_info = {"sex": ["M", "F", "I"]}
        to_features.set_categorical_info(cat_column_info)
        to_features.add_chunk(df)
        to_features.add_chunk(df)
        ...
        to_features.finalize()
        ```
    """

    def __init__(self, output_uuid: str, id_column=None, event_time_column=None):
        """PandasToFeaturesChunked Initialization"""

        # Make sure the output_uuid is a valid name
        Artifact.is_name_valid(output_uuid)

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.first_chunk = None
        self.pandas_to_features = PandasToFeatures(output_uuid)

    def set_categorical_info(self, cat_column_info: dict[list[str]]):
        """Set the Categorical Columns
        Args:
            cat_column_info (dict[list[str]]): Dictionary of categorical columns and their possible values
        """

        # Create the CategoricalDtypes
        cat_d_types = {}
        for col, vals in cat_column_info.items():
            cat_d_types[col] = CategoricalDtype(categories=vals)

        # Now set the CategoricalDtypes on our underlying PandasToFeatures
        self.pandas_to_features.categorical_dtypes = cat_d_types

    def add_chunk(self, chunk_df: pd.DataFrame):
        """Add a Chunk of Data to the FeatureSet"""

        # Is this the first chunk? If so we need to run the pre_transform
        if self.first_chunk is None:
            self.log.info(f"Adding first chunk {chunk_df.shape}...")
            self.first_chunk = chunk_df
            self.pandas_to_features.set_input(chunk_df, self.id_column, self.event_time_column)
            self.pandas_to_features.pre_transform()
            self.pandas_to_features.transform_impl()
        else:
            self.log.info(f"Adding chunk {chunk_df.shape}...")
            self.pandas_to_features.set_input(chunk_df, self.id_column, self.event_time_column)
            self.pandas_to_features.transform_impl()

    def pre_transform(self, **kwargs):
        """Pre-Transform: Create the Feature Group with Chunked Data"""

        # Loading data into a Feature Group takes a while, so set status to loading
        FeatureSetCore(self.output_uuid).set_status("loading")

    def transform_impl(self):
        """Required implementation of the Transform interface"""
        self.log.warning("PandasToFeaturesChunked.transform_impl() called.  This is a no-op.")

    def post_transform(self, **kwargs):
        """Post-Transform: Any Post Transform Steps"""
        self.pandas_to_features.post_transform()


if __name__ == "__main__":
    """Exercise the PandasToFeaturesChunked Class"""
    from workbench.api.data_source import DataSource

    # Load in a DataFrame from a DataSource and split it into chunks
    ds = DataSource("abalone_data")
    df = ds.query("select * from abalone_data")
    print(f"df.shape: {df.shape}")

    # Build DataFrames chunks of 1000 rows each
    chunk_size = 1000
    chunks = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    # Create our PandasToFeaturesChunked class
    to_features = PandasToFeaturesChunked("abalone_features")

    # Manually set the Categorical Columns
    categorical_column_info = {"sex": ["M", "F", "I"]}
    to_features.set_categorical_info(categorical_column_info)

    # Now loop through the chunks and add them to the FeatureSet
    for chunk in chunks:
        to_features.add_chunk(chunk)
    to_features.finalize()
