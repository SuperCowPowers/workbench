"""DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking"""

import awswrangler as wr

# Local imports
from sageworks.core.transforms.transform import Transform
from sageworks.core.transforms.pandas_transforms.pandas_to_features_chunked import (
    PandasToFeaturesChunked,
)
from sageworks.core.artifacts.data_source_factory import DataSourceFactory


class DataToFeaturesChunk(Transform):
    """DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking

    Common Usage:
        ```
        data_to_features = DataToFeaturesChunk(input_uuid, output_uuid, 50000)
        data_to_features.set_output_tags(["heavy", "whatever"])
        data_to_features.transform(query, id_column, event_time_column=None)
        ```
    """

    def __init__(self, input_uuid: str, output_uuid: str, chunk_size: int = 50000):
        """DataToFeaturesChunk Initialization"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.chunk_size = chunk_size
        self.input_data_source = DataSourceFactory(input_uuid)
        self.ds_database = "sageworks"
        self.cat_column_info = {}
        self.chunked_to_features = None

    def set_categorical_info(self, cat_column_info: dict[list[str]]):
        """Set the Categorical Columns Information
        Args:
            cat_column_info (dict[list[str]]): Dictionary of categorical columns and their possible values
        """
        self.cat_column_info = cat_column_info

    def detect_categoral_columns(self):
        """Automatically figure out which columns are categorical"""
        self.log.info("Automatically computing the categorical columns")

        # DataSources will compute value_counts for each object/str column
        value_counts = self.input_data_source.value_counts()
        for column, value_info in value_counts.items():
            # Hack to avoid IP columns
            if column.endswith("ip"):
                self.log.info(f"Skipping column {column}...")
                continue
            # How many unique values are there?
            unique = len(value_info.keys())
            if 1 < unique < 6:
                self.log.info(f"Column {column} is categorical (unique={unique})")
                self.cat_column_info[column] = list(value_info.keys())

    def pre_transform(self, **kwargs):
        """Figure out which fields are categorical"""

        # If the user didn't specify any categorical columns, try to figure them out
        if not self.cat_column_info:
            self.detect_categoral_columns()

    def transform_impl(self, query, id_column: str, event_time_column: str = None):
        """Convert the Data Source into a Feature Set using Chunking"""

        # Create our PandasToFeaturesChunked class
        self.chunked_to_features = PandasToFeaturesChunked(self.output_uuid, id_column, event_time_column)
        self.chunked_to_features.set_output_tags(self.output_tags)
        self.chunked_to_features.set_categorical_info(self.cat_column_info)
        self.chunked_to_features.pre_transform()

        # Read in the data from Athena in chunks
        for chunk in wr.athena.read_sql_query(
            query,
            database=self.ds_database,
            ctas_approach=False,
            chunksize=self.chunk_size,
            boto3_session=self.boto3_session,
        ):
            # Hand off each chunk of data
            self.chunked_to_features.add_chunk(chunk)

    def post_transform(self, **kwargs):
        """Post-Transform: Any Post Transform Steps"""
        self.log.info("Post-Transform: Completing FeatureSet Offline Storage...")
        self.chunked_to_features.post_transform()


if __name__ == "__main__":
    """Exercise the DataToFeaturesChunk Class"""

    # Create my DF to Feature Set Transform
    data_to_features = DataToFeaturesChunk("http_10795", "http_features")
    data_to_features.set_output_tags(["http", "nomic"])

    # Store this dataframe as a SageWorks Feature Set

    # Old Stuff
    """
    fields = [
        "timestamp",
        "flow_id",
        "in_iface",
        "proto",
        "dns_type",
        "dns_rrtype",
        "dns_flags",
        "dns_rcode",
    ]
    query = f"SELECT {', '.join(fields)} FROM heavy_dns limit 1000"
    """
    query = "SELECT * FROM http_10795"
    data_to_features.transform(query=query, id_column="flow_id", event_time_column="timestamp")
