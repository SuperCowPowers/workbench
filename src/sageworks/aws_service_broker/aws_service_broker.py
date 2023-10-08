"""AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
import os
import sys
import argparse
from enum import Enum, auto
import logging
from threading import Thread
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.utils.cache import Cache
from sageworks.utils.redis_cache import RedisCache
from sageworks.aws_service_broker.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_broker.aws_service_connectors.glue_jobs import GlueJobs
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog
from sageworks.aws_service_broker.aws_service_connectors.feature_store import (
    FeatureStore,
)
from sageworks.aws_service_broker.aws_service_connectors.model_registry import (
    ModelRegistry,
)
from sageworks.aws_service_broker.aws_service_connectors.endpoints import Endpoints
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger("sageworks")


# Enumerated types for SageWorks Meta Requests
class ServiceCategory(Enum):
    """Enumerated Types for SageWorks Meta Requests"""

    INCOMING_DATA_S3 = auto()
    GLUE_JOBS = auto()
    DATA_SOURCES_S3 = auto()
    FEATURE_SETS_S3 = auto()
    DATA_CATALOG = auto()
    FEATURE_STORE = auto()
    MODELS = auto()
    ENDPOINTS = auto()


class AWSServiceBroker:
    """AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""

    # Note: This database_scope is a list of databases that we want to pull metadata from
    #       At some point, we should pull this from a config file.
    database_scope = ["sageworks", "sagemaker_featurestore"]

    def __new__(cls):
        """AWSServiceBroker Singleton Pattern"""
        if not hasattr(cls, "instance"):
            log.debug("Creating the AWSServiceBroker Singleton...")
            cls.instance = super(AWSServiceBroker, cls).__new__(cls)

            # Class Initialization
            cls.instance.__class_init__(cls.database_scope)

        return cls.instance

    @classmethod
    def __class_init__(cls, database_scope):
        """AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
        cls.log = logging.getLogger("sageworks")

        # Grab our SageWorks Bucket
        cls.sageworks_bucket = os.environ.get("SAGEWORKS_BUCKET")
        if cls.sageworks_bucket is None:
            print("Could not find ENV var for SAGEWORKS_BUCKET!")
            sys.exit(1)

        # Construct bucket paths
        cls.incoming_data_bucket = "s3://" + cls.sageworks_bucket + "/incoming-data/"
        cls.data_sources_bucket = "s3://" + cls.sageworks_bucket + "/data-sources/"
        cls.feature_sets_bucket = "s3://" + cls.sageworks_bucket + "/feature-sets/"

        # SageWorks category mapping to AWS Services
        # - incoming_data = S3
        # - data_sources = Data Catalog, Athena
        # - feature_sets = Data Catalog, Athena, Feature Store
        # - models = Model Registry
        # - endpoints = Sagemaker Endpoints, Model Monitors

        # Pull in AWS Service Connectors
        cls.incoming_data_s3 = S3Bucket(cls.incoming_data_bucket)
        cls.data_sources_s3 = S3Bucket(cls.data_sources_bucket)
        cls.feature_sets_s3 = S3Bucket(cls.feature_sets_bucket)
        cls.glue_jobs = GlueJobs()
        cls.data_catalog = DataCatalog(database_scope)
        cls.feature_store = FeatureStore()
        cls.model_registry = ModelRegistry()
        cls.endpoints = Endpoints()

        # Cache for Metadata
        if RedisCache().check():
            cls.meta_cache = RedisCache()
            cls.fresh_cache = RedisCache(expire=20, postfix=":fresh")
            cls.open_threads = []
        else:
            cls.meta_cache = Cache()
            cls.fresh_cache = Cache(expire=20)
            cls.open_threads = []

        # This connection map sets up the connector objects for each category of metadata
        # Note: Even though this seems confusing, it makes other code WAY simpler
        cls.connection_map = {
            ServiceCategory.INCOMING_DATA_S3: cls.incoming_data_s3,
            ServiceCategory.GLUE_JOBS: cls.glue_jobs,
            ServiceCategory.DATA_SOURCES_S3: cls.data_sources_s3,
            ServiceCategory.FEATURE_SETS_S3: cls.feature_sets_s3,
            ServiceCategory.DATA_CATALOG: cls.data_catalog,
            ServiceCategory.FEATURE_STORE: cls.feature_store,
            ServiceCategory.MODELS: cls.model_registry,
            ServiceCategory.ENDPOINTS: cls.endpoints,
        }

    @classmethod
    def refresh_aws_data(cls, category: ServiceCategory) -> None:
        """Refresh the Metadata for the given Category

        Args:
            category (ServiceCategory): The Category of metadata to Pull
        """
        # Refresh the connection for the given category and pull new data
        try:
            cls.connection_map[category].refresh()
            cls.meta_cache.set(category, cls.connection_map[category].aws_meta())
            cls.fresh_cache.set(category, True)
        except ClientError as e:
            cls.log.warning(f"Failed to refresh AWS data for {category}: {e}")

    @classmethod
    def get_metadata(cls, category: ServiceCategory, force_refresh: bool = False) -> dict:
        """Pull Metadata for the given Service Category

        Args:
            category (ServiceCategory): The Service Category to pull metadata from
            force_refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: The Metadata for the Requested Service Category
        """
        # Logic:
        # - NO metadata in the cache, we need to BLOCK and get it
        # - Metadata is in the cache but is stale, launch a thread to refresh, return stale data
        # - Metadata is in the cache and is fresh, so we just return it

        # Do we have this AWS data already in the cache?
        meta_data = cls.meta_cache.get(category)

        # If we don't have the AWS data in the cache, we need to BLOCK and get it
        if meta_data is None or force_refresh:
            cls.log.info(f"Blocking: Getting metadata for {category}...")
            cls.refresh_aws_data(category)
            return cls.meta_cache.get(category)

        # Is the AWS data stale?
        if cls.fresh_cache.get(category) is None:
            cls.log.debug(f"Async: Metadata for {category} is stale, launching refresh thread...")
            cls.fresh_cache.set(category, True)
            thread = Thread(target=cls.refresh_aws_data, args=(category,))
            cls.open_threads.append(thread)
            thread.start()
            return cls.meta_cache.get(category)

        # If the metadata is fresh, just return it
        cls.log.debug(f"Metadata for {category} is fresh!")
        return cls.meta_cache.get(category)

    @classmethod
    def get_all_metadata(cls, force_refresh=False) -> dict:
        """Pull the metadata for ALL the Service Categories
        Args:
            force_refresh (bool, optional): Force a refresh of the metadata. Defaults to False.
        Returns:
            dict: The Metadata for ALL the Service Categories
        """
        cls.log.debug("Getting ALL AWS Metadata: You should call get_metadata() with specific categories")
        return {_category: cls.get_metadata(_category, force_refresh) for _category in ServiceCategory}

    @classmethod
    def wait_for_refreshes(cls) -> None:
        """Wait for any open threads to finish"""
        for thread in cls.open_threads:
            thread.join()

    @classmethod
    def get_s3_object_sizes(cls, category: ServiceCategory, prefix: str = "") -> int:
        """Get the total size of all the objects in the given S3 Prefix
        Args:
            category (ServiceCategory): Can be either INCOMING_DATA_S3, DATA_SOURCES_S3, or FEATURE_SETS_S3
            prefix (str): The S3 Prefix, all files under this prefix will be summed up
        Returns:
            int: The total size of all the objects in the given S3 Prefix
        """
        # Get the metadata for the given category
        meta_data = cls.get_metadata(category)

        # Get the total size of all the objects in the given S3 Prefix
        total_size = 0
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        for file, info in meta_data.items():
            if prefix in file:
                total_size += info["ContentLength"]
        return total_size


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

    # Get the Metadata for ALL the categories
    # NOTE: This should force a refresh of the metadata
    # pprint(aws_broker.get_all_metadata(force_refresh=True))

    # Get S3 object sizes
    incoming_data_size = aws_broker.get_s3_object_sizes(ServiceCategory.INCOMING_DATA_S3)
    print(f"Incoming Data Size: {incoming_data_size} Bytes")

    data_sources_size = aws_broker.get_s3_object_sizes(ServiceCategory.DATA_SOURCES_S3)
    print(f"Data Sources Size: {data_sources_size} Bytes")

    abalone_size = aws_broker.get_s3_object_sizes(ServiceCategory.DATA_SOURCES_S3, prefix="abalone_data")
    print(f"Abalone Size: {abalone_size} Bytes")

    # Wait for any open threads to finish
    aws_broker.wait_for_refreshes()
