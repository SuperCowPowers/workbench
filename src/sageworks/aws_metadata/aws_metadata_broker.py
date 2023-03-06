"""AWSMetadataBroker pulls and collects metadata from a bunch of AWS Services"""
import sys
import argparse
from enum import Enum, auto
import logging

# Local Imports
from sageworks.aws_metadata.cache import Cache
from sageworks.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_connectors.data_catalog import DataCatalog
from sageworks.aws_service_connectors.feature_store import FeatureStore
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


# Enumerated types for SageWorks Meta Requests
class MetaCategory(Enum):
    """Enumerated Types for SageWorks Meta Requests"""
    INCOMING_DATA = auto()
    DATA_SOURCES = auto()
    FEATURE_SETS = auto()
    MODELS = auto()
    ENDPOINTS = auto()


class AWSMetadataBroker:

    def __new__(cls, database_scope='sageworks'):
        """AWSMetadataBroker Singleton Pattern"""
        if not hasattr(cls, 'instance'):
            print('Creating New AWSMetadataBroker Instance...')
            cls.instance = super(AWSMetadataBroker, cls).__new__(cls)

            # Class Initialization
            cls.instance.__class_init__(database_scope)

        return cls.instance

    @classmethod
    def __class_init__(cls, database_scope='sageworks'):
        """"AWSMetadataBroker pulls and collects metadata from a bunch of AWS Services"""
        cls.log = logging.getLogger(__file__)

        # FIXME: This should be pulled from a config file
        cls.incoming_data_bucket = 's3://sageworks-incoming-data'

        # SageWorks category mapping to AWS Services
        # - incoming_data = S3
        # - data_sources = Data Catalog, Athena
        # - feature_sets = Data Catalog, Athena, Feature Store
        # - models = Model Registry
        # - endpoints = Sagemaker Endpoints, Model Monitors

        # Pull in AWS Service Connectors
        cls.incoming_data = S3Bucket(cls.incoming_data_bucket)
        cls.data_catalog = DataCatalog(database_scope)
        cls.feature_store = FeatureStore()
        # Model Registry
        # Endpoints
        # Model Monitors

        # Temporal Cache for Metadata
        cls.meta_cache = Cache(10)

        # This connection map sets up the connector objects for each category of metadata
        # Note: Even though this seems confusing, it makes other code WAY simpler
        cls.connection_map = {
            MetaCategory.INCOMING_DATA: cls.incoming_data,
            MetaCategory.DATA_SOURCES: cls.data_catalog,
            MetaCategory.FEATURE_SETS: cls.feature_store,
            MetaCategory.MODELS: cls.incoming_data,
            MetaCategory.ENDPOINTS: cls.incoming_data
        }

    @classmethod
    def refresh_meta(cls, category: MetaCategory) -> None:
        """Refresh the Metadata for the given Category

        Args:
            category (MetaCategory): The Category of metadata to Pull
        """
        # Refresh the connection for the given category and pull new data
        cls.connection_map[category].refresh()
        cls.meta_cache.set(category, cls.connection_map[category].metadata())

    @classmethod
    def get_metadata(cls, category: MetaCategory) -> dict:
        """Pull Metadata for the given Category

        Args:
            category (MetaCategory): The Category of metadata to Pull

        Returns:
            dict: The Metadata for the Requested Category
        """
        # Check the Temporal Cache
        meta_data = cls.meta_cache.get(category)
        if meta_data is not None:
            return meta_data
        else:
            # If we don't have the data in the cache, we need to refresh it
            cls.log.info(f"Refreshing data for {category}...")
            cls.refresh_meta(category)
            return cls.meta_cache.get(category)


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='sageworks', help='AWS Data Catalog Database')
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class
    meta_broker = AWSMetadataBroker('all')

    # Get the Metadata for various categories
    for my_category in MetaCategory:
        print(f"{my_category}:")
        pprint(meta_broker.get_metadata(my_category))

    # Get the Metadata for various categories
    # NOTE: There should be NO Refreshes in the logs
    for my_category in MetaCategory:
        print(f"{my_category}:")
        pprint(meta_broker.get_metadata(my_category))
