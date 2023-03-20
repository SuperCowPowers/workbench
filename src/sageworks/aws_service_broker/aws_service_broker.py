"""AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
import sys
import argparse
from enum import Enum, auto
import logging

# SageWorks Imports
from sageworks.aws_service_broker.cache import Cache
from sageworks.aws_service_broker.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog
from sageworks.aws_service_broker.aws_service_connectors.feature_store import FeatureStore
from sageworks.aws_service_broker.aws_service_connectors.model_registry import ModelRegistry
from sageworks.aws_service_broker.aws_service_connectors.endpoints import Endpoints
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


# Enumerated types for SageWorks Meta Requests
class ServiceCategory(Enum):
    """Enumerated Types for SageWorks Meta Requests"""
    INCOMING_DATA = auto()
    DATA_CATALOG = auto()
    FEATURE_STORE = auto()
    MODELS = auto()
    ENDPOINTS = auto()


class AWSServiceBroker:

    def __new__(cls, database_scope=['sageworks', 'sagemaker_featurestore']):
        """AWSServiceBroker Singleton Pattern"""
        if not hasattr(cls, 'instance'):
            print('Creating New AWSServiceBroker Instance...')
            cls.instance = super(AWSServiceBroker, cls).__new__(cls)

            # Class Initialization
            cls.instance.__class_init__(database_scope)

        return cls.instance

    @classmethod
    def __class_init__(cls, database_scope):
        """"AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
        cls.log = logging.getLogger(__file__)

        # FIXME: This should be pulled from a config file
        cls.incoming_data_bucket = 's3://scp-sageworks-artifacts/incoming-data'

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
        cls.model_registry = ModelRegistry()
        cls.endpoints = Endpoints()
        # Model Monitors

        # Temporal Cache for Metadata
        cls.meta_cache = Cache(timeout=10)

        # This connection map sets up the connector objects for each category of metadata
        # Note: Even though this seems confusing, it makes other code WAY simpler
        cls.connection_map = {
            ServiceCategory.INCOMING_DATA: cls.incoming_data,
            ServiceCategory.DATA_CATALOG: cls.data_catalog,
            ServiceCategory.FEATURE_STORE: cls.feature_store,
            ServiceCategory.MODELS: cls.model_registry,
            ServiceCategory.ENDPOINTS: cls.endpoints
        }

    @classmethod
    def refresh_meta(cls, category: ServiceCategory) -> None:
        """Refresh the Metadata for the given Category

        Args:
            category (ServiceCategory): The Category of metadata to Pull
        """
        # Refresh the connection for the given category and pull new data
        cls.connection_map[category].refresh()
        cls.meta_cache.set(category, cls.connection_map[category].metadata())

    @classmethod
    def get_metadata(cls, category: ServiceCategory) -> dict:
        """Pull Metadata for the given Service Category

        Args:
            category (ServiceCategory): The Service Category to pull metadata from

        Returns:
            dict: The Metadata for the Requested Service Category
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

    @classmethod
    def get_all_metadata(cls) -> dict:
        """Pull the metadata for ALL the Service Categories"""
        return {_category: cls.get_metadata(_category) for _category in ServiceCategory}


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
    aws_broker = AWSServiceBroker()

    # Get the Metadata for various categories
    for my_category in ServiceCategory:
        print(f"{my_category}:")
        pprint(aws_broker.get_metadata(my_category))

    # Get the Metadata for ALL the categories
    # NOTE: There should be NO Refreshes in the logs
    pprint(aws_broker.get_all_metadata())
