"""AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
import sys
import argparse
from enum import Enum, auto
import logging
from threading import Thread

# SageWorks Imports
from sageworks.utils.cache import Cache
from sageworks.utils.redis_cache import RedisCache
from sageworks.aws_service_broker.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog
from sageworks.aws_service_broker.aws_service_connectors.feature_store import FeatureStore
from sageworks.aws_service_broker.aws_service_connectors.model_registry import ModelRegistry
from sageworks.aws_service_broker.aws_service_connectors.endpoints import Endpoints
from sageworks.aws_service_broker.aws_service_connectors.artifact_info import ArtifactInfo
from sageworks.utils.sageworks_config import SageWorksConfig
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
    # Note: This database_scope is a list of databases that we want to pull metadata from
    #       At some point, we should pull this from a config file.
    database_scope = ["sageworks", "sagemaker_featurestore"]

    def __new__(cls):
        """AWSServiceBroker Singleton Pattern"""
        if not hasattr(cls, "instance"):
            print("Creating New AWSServiceBroker Instance...")
            cls.instance = super(AWSServiceBroker, cls).__new__(cls)

            # Class Initialization
            cls.instance.__class_init__(cls.database_scope)

        return cls.instance

    @classmethod
    def __class_init__(cls, database_scope):
        """AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
        cls.log = logging.getLogger(__file__)

        # Grab our SageWorksConfig for S3 Buckets and other SageWorks specific settings
        sageworks_config = SageWorksConfig()
        sageworks_bucket = sageworks_config.get_config_value("SAGEWORKS_AWS", "S3_BUCKET_NAME")
        cls.incoming_data_bucket = "s3://" + sageworks_bucket + "/incoming-data"

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
        cls.artifact_info = ArtifactInfo()

        # Model Monitors

        # Cache for Metadata
        if RedisCache().check():
            cls.meta_cache = RedisCache()
            cls.fresh_cache = RedisCache(expire=10, postfix=":fresh")
            cls.open_threads = []
        else:
            cls.meta_cache = Cache()
            cls.fresh_cache = Cache(expire=10)
            cls.open_threads = []

        # This connection map sets up the connector objects for each category of metadata
        # Note: Even though this seems confusing, it makes other code WAY simpler
        cls.connection_map = {
            ServiceCategory.INCOMING_DATA: cls.incoming_data,
            ServiceCategory.DATA_CATALOG: cls.data_catalog,
            ServiceCategory.FEATURE_STORE: cls.feature_store,
            ServiceCategory.MODELS: cls.model_registry,
            ServiceCategory.ENDPOINTS: cls.endpoints,
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
        cls.fresh_cache.set(category, True)

    @classmethod
    def get_metadata(cls, category: ServiceCategory) -> dict:
        """Pull Metadata for the given Service Category

        Args:
            category (ServiceCategory): The Service Category to pull metadata from

        Returns:
            dict: The Metadata for the Requested Service Category
        """
        # Logic:
        # - NO metadata in the cache, we need to BLOCK and get it
        # - Metadata is in the cache but is stale, launch a thread to refresh, return stale data
        # - Metadata is in the cache and is fresh, so we just return it

        # Do we have the metadata in the cache?
        meta_data = cls.meta_cache.get(category)

        # If we don't have the metadata in the cache, we need to BLOCK and get it
        if meta_data is None:
            cls.log.info(f"Blocking: Getting metadata for {category}...")
            cls.refresh_meta(category)
            return cls.meta_cache.get(category)

        # Is the metadata stale?
        if cls.fresh_cache.get(category) is None:
            cls.log.info(f"Async: Metadata for {category} is stale, launching refresh thread...")
            thread = Thread(target=cls.refresh_meta, args=(category,))
            cls.open_threads.append(thread)
            thread.start()
            return cls.meta_cache.get(category)

        # If the metadata is fresh, just return it
        cls.log.info(f"Metadata for {category} is fresh!")
        return cls.meta_cache.get(category)

    @classmethod
    def get_all_metadata(cls, warn=True) -> dict:
        """Pull the metadata for ALL the Service Categories"""
        if warn:
            cls.log.warning("Getting ALL AWS Metadata: You should call get_metadata() with specific categories")
        return {_category: cls.get_metadata(_category) for _category in ServiceCategory}

    @classmethod
    def wait_for_refreshes(cls) -> None:
        """Wait for any open threads to finish"""
        for thread in cls.open_threads:
            thread.join()


if __name__ == "__main__":
    """Exercise the AWS Service Broker Class"""
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="sageworks", help="AWS Data Catalog Database")
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
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

    # Get S3 object sizes
    incoming_data_size = aws_broker.artifact_info.s3_object_sizes(aws_broker.incoming_data.bucket)
    print(f"Incoming Data Size: {incoming_data_size} MB")

    # Wait for any open threads to finish
    aws_broker.wait_for_refreshes()
