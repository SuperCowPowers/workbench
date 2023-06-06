"""DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking"""
import awswrangler as wr

# Local imports
from sageworks.transforms.transform import Transform
from sageworks.transforms.pandas_transforms.pandas_to_features_chunked import PandasToFeaturesChunked
from sageworks.artifacts.data_sources.data_source import DataSource


class DataToFeaturesChunk(Transform):
    """DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking

    Common Usage:
        to_features = DataToFeaturesChunk(input_uuid, output_uuid, 50000)
        to_features.set_output_tags(["heavy", "whatever"])
        to_features.transform(query, id_column, event_time_column=None)
    """

    def __init__(self, input_uuid: str, output_uuid: str, chunk_size: int = 50000):
        """DataToFeaturesChunk Initialization"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.chunk_size = chunk_size
        self.input_data_source = DataSource(input_uuid)
        self.ds_database = "sageworks"
        self.cat_column_info = {}

    def pre_transform(self, **kwargs):
        """Figure out which fields are categorical"""

        self.log.info("Precomputing the categorical columns")

        # DataSources will compute value_counts for each object/str column
        value_counts = self.input_data_source.value_counts()
        for column, value_info in value_counts.items():
            # How many unique values are there?
            unique = len(value_info.keys())
            if 1 < unique < 10:
                self.log.info(f"Column {column} is categorical (unique={unique})")
                self.cat_column_info[column] = list(value_info.keys())

    def transform_impl(self, query, id_column: str, event_time_column: str = None):
        """Convert the Data Source into a Feature Set using Chunking"""

        # Create our PandasToFeaturesChunked class
        to_features = PandasToFeaturesChunked(self.output_uuid, id_column, event_time_column)
        to_features.set_output_tags(self.output_tags)
        to_features.set_categorical_info(self.cat_column_info)

        # Read in the data from Athena in chunks
        for chunk in wr.athena.read_sql_query(
            query,
            database=self.ds_database,
            ctas_approach=False,
            chunksize=self.chunk_size,
            boto3_session=self.boto_session,
        ):
            # Hand off each chunk of data
            to_features.add_chunk(chunk)

        # Finalize the FeatureSet
        to_features.finalize()


if __name__ == "__main__":
    """Exercise the DataToFeaturesChunk Class"""

    # Create my DF to Feature Set Transform
    data_to_features = DataToFeaturesChunk("heavy_dns", "dns_features_test", 100)
    data_to_features.set_output_tags(["test", "heavy"])

    # Store this dataframe as a SageWorks Feature Set
    fields = ["timestamp", "flow_id", "in_iface", "proto", "dns_type", "dns_rrtype", "dns_flags", "dns_rcode"]
    query = f"SELECT {', '.join(fields)} FROM heavy_dns limit 1000"
    data_to_features.transform(query=query, id_column="flow_id", event_time_column="timestamp")
