"""FeatureSet: SageWork Feature Set accessible through Athena"""
import logging

# Local Imports
from sageworks.artifacts.data_sources.athena_source import AthenaSource
from sageworks.aws_metadata.aws_metadata_broker import MetaCategory, AWSMetadataBroker
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class FeatureSet(AthenaSource):

    def __init__(self, feature_set_name):
        """FeatureSet: SageWork Feature Set accessible through Athena

        Args:
            feature_set_name (str): Name of Feature Set in SageWorks Metadata.
        """
        self.log = logging.getLogger(__name__)
        self.feature_set_name = feature_set_name

        # Grab an AWS Metadata Broker object and pull information for Feature Sets
        self.aws_meta = AWSMetadataBroker('all')
        self.feature_set_meta = self.aws_meta.get_metadata(MetaCategory.FEATURE_SETS).get(self.feature_set_name)

        # Pull Athena and S3 Storage information from metadata
        self.athena_database = self.feature_set_meta['sageworks'].get('athena_database')
        self.athena_table = self.feature_set_meta['sageworks'].get('athena_table')
        self.s3_storage = self.feature_set_meta['sageworks'].get('s3_storage')

        # Call SuperClass (AthenaSource) Initialization
        super().__init__(self.athena_database, self.athena_table)

        # All done
        print(f'FeatureSet Initialized: {feature_set_name}')

    def check(self) -> bool:
        """Does the feature_set_name exist in the SageWorks Metadata?"""
        if self.feature_set_meta is None:
            self.log.critical(f'FeatureSet.check() {self.feature_set_name} not found in SageWorks Metadata!')
            return False
        return True


# Simple test of the FeatureSet functionality
def test():
    """Test for FeatureSet Class"""

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet('AqSolDB-base')

    # Call the various methods

    # Does this Feature Set exist?
    assert(my_features.check())

    # How many rows and columns?
    num_rows = my_features.get_num_rows()
    num_columns = my_features.get_num_columns()
    print(f'Rows: {num_rows} Columns: {num_columns}')

    # What are the column names?
    columns = my_features.get_column_names()
    print(columns)

    # Get the tags associated with this feature set
    print(f"Tags: {my_features.get_tags()}")

    # Run a query to only pull back a few columns and rows
    column_query = ', '.join(columns[:3])
    query = f'select {column_query} from "{my_features.database_name}"."{my_features.table_name}" limit 10'
    df = my_features.query(query)
    print(df)


if __name__ == "__main__":
    test()
