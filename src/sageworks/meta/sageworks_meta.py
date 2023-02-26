"""SageWorksMeta pulls and collects metadata from a bunch of AWS Services"""
import sys
import argparse
from enum import Enum, auto
import logging

# Local Imports
from sageworks.meta.cache import Cache
from sageworks.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_connectors.data_catalog import DataCatalog
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


class SageWorksMeta:
    """SageWorksMeta pulls and collects metadata from a bunch of AWS Services"""
    def __init__(self):
        """SageWorksMeta Initialization"""
        self.log = logging.getLogger(__file__)

        # FIXME: These should be pulled from a config file
        self.incoming_data_bucket = 's3://sageworks-incoming-data'
        self.data_catalog_db = 'sageworks'

        # SageWorks category mapping to AWS Services
        # - incoming_data = S3
        # - data_sources = Data Catalog, Athena
        # - feature_sets = Athena, Feature Store
        # - models = Model Registry
        # - endpoints = Sagemaker Endpoints, Model Monitors

        # Pull in AWS Service Connectors
        self.incoming_data = S3Bucket(self.incoming_data_bucket)
        self.data_catalog = DataCatalog(self.data_catalog_db)
        self.meta_cache = Cache(10)
        # Feature Store
        # Model Registry
        # Endpoints
        # Model Monitors

        # This connection map is used to refresh the metadata for each category
        # Note: Even though this seems confusing, it makes other code WAY simpler
        self.connection_map = {
            MetaCategory.INCOMING_DATA: {'refresh': self.incoming_data.refresh, 'get_data': self.incoming_data.get_data},
            MetaCategory.DATA_SOURCES: {'refresh': self.data_catalog.refresh, 'get_data': self.data_catalog.get_data},
            MetaCategory.FEATURE_SETS: {'refresh': lambda: None, 'get_data': lambda: []},
            MetaCategory.MODELS: {'refresh': lambda: None, 'get_data': lambda: []},
            MetaCategory.ENDPOINTS: {'refresh': lambda: None, 'get_data': lambda: []}
        }

    def refresh_meta(self, category: MetaCategory) -> None:
        """Refresh the Metadata for the given Category

        Args:
            category (MetaCategory): The Category of metadata to Pull
        """
        # Refresh the connection for the given category and pull new data
        self.connection_map[category]['refresh']()
        self.meta_cache.set(category, self.connection_map[category]['get_data']())

    def get(self, category: MetaCategory) -> dict:
        """Pull Metadata for the given Category

        Args:
            category (MetaCategory): The Category of metadata to Pull

        Returns:
            dict: The Metadata for the Requested Category
        """
        # Check the Temporal Cache
        meta_data = self.meta_cache.get(category)
        if meta_data is not None:
            return meta_data
        else:
            # If we don't have the data in the cache, we need to refresh it
            self.log.info(f"Refreshing data for {category}...")
            self.refresh_meta(category)
            return self.meta_cache.get(category)


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
    sage_meta = SageWorksMeta()

    # Get the Metadata for various categories
    for my_category in MetaCategory:
        print(f"{my_category}:")
        pprint(sage_meta.get(my_category))

    # Get the Metadata for various categories
    # NOTE: There should be NO Refreshes in the logs
    for my_category in MetaCategory:
        print(f"{my_category}:")
        pprint(sage_meta.get(my_category))
