"""AWSServiceBroker pulls and collects metadata from a bunch of AWS Services
   Note: This is the most complicated/messy class in the entire SageWorks project.
      If you're looking at the class you're probably in the wrong place.
      We suggest looking at these classes instead:
      - api/meta.py (API for the data produced by this class)
      - aws_service_broker/aws_service_connectors/*.py (Pulls the data from AWS)
"""

import sys
import time
import argparse
from enum import Enum, auto
import logging
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.utils.config_manager import ConfigManager
from sageworks.aws_service_broker.aws_service_connectors.s3_bucket import S3Bucket
from sageworks.aws_service_broker.aws_service_connectors.glue_jobs import GlueJobs
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog
from sageworks.aws_service_broker.aws_service_connectors.feature_store import FeatureStore
from sageworks.aws_service_broker.aws_service_connectors.model_registry import ModelRegistry
from sageworks.aws_service_broker.aws_service_connectors.endpoints import Endpoints
from sageworks.utils.sageworks_cache import SageWorksCache


# Enumerated types for SageWorks Meta Requests
class ServiceCategory(Enum):
    """Enumerated Types for SageWorks Meta Requests"""

    INCOMING_DATA_S3 = auto()
    GLUE_JOBS = auto()
    # DATA_SOURCES_S3 = auto()  Memory Tests
    # FEATURE_SETS_S3 = auto()  Memory Tests
    DATA_CATALOG = auto()
    FEATURE_STORE = auto()
    MODELS = auto()
    ENDPOINTS = auto()


class AWSServiceBroker:
    """AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""

    def __new__(cls):
        """AWSServiceBroker Singleton Pattern"""
        if not hasattr(cls, "instance"):
            # Initialize class attributes here
            cls.log = logging.getLogger("sageworks")
            cls.database_scope = ["sageworks", "sagemaker_featurestore"]
            cls.log.info("Creating the AWSServiceBroker Singleton...")
            cls.instance = super(AWSServiceBroker, cls).__new__(cls)

            # Class Initialization
            cls.instance.__class_init__()

        return cls.instance

    @classmethod
    def __class_init__(cls):
        """AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""

        # Grab our SageWorks Bucket
        cm = ConfigManager()
        cls.sageworks_bucket = cm.get_config("SAGEWORKS_BUCKET")

        # Sanity check our SageWorks Bucket
        if cls.sageworks_bucket is None:
            cls.log.critical("SAGEWORKS_BUCKET is not defined")
            cls.log.critical("Run Initial Setup here: https://supercowpowers.github.io/sageworks/getting_started/")
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
        # cls.data_sources_s3 = S3Bucket(cls.data_sources_bucket)  Memory Tests
        # cls.feature_sets_s3 = S3Bucket(cls.feature_sets_bucket)  Memory Tests
        cls.glue_jobs = GlueJobs()
        cls.data_catalog = DataCatalog(cls.database_scope)
        cls.feature_store = FeatureStore()
        cls.model_registry = ModelRegistry()
        cls.endpoints = Endpoints()

        # Caches for Metadata
        cls.meta_cache = SageWorksCache()
        cls.fresh_cache = SageWorksCache(expire=60, postfix=":fresh")

        # Thread Pool for Refreshes
        # Note: We only need 1 thread for data refreshes, a bunch of threads = AWS Throttling
        cls.thread_pool = ThreadPoolExecutor(max_workers=1)

        # This connection map sets up the connector objects for each category of metadata
        # Note: Even though this seems confusing, it makes other code WAY simpler
        cls.connection_map = {
            ServiceCategory.INCOMING_DATA_S3: cls.incoming_data_s3,
            ServiceCategory.GLUE_JOBS: cls.glue_jobs,
            # ServiceCategory.DATA_SOURCES_S3: cls.data_sources_s3,  Memory Tests
            # ServiceCategory.FEATURE_SETS_S3: cls.feature_sets_s3,  Memory Tests
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
        sleep_times = [1, 2, 4, 8, 16, 32, 64]
        max_attempts = len(sleep_times)
        for attempt in range(max_attempts):
            try:
                cls.fresh_cache.set(category, True)
                cls.connection_map[category].refresh()
                cls.meta_cache.set(category, cls.connection_map[category].summary())
                return  # Success, exit the loop
            except ClientError as error:
                error_code = error.response["Error"]["Code"]
                error_message = error.response["Error"]["Message"]
                cls.log.warning(
                    f"Attempt {attempt}: Failed to refresh AWS data for {category}: {error_code} - {error_message}"
                )

                # Exponential backoff for ThrottlingExceptions
                if error_code == "ThrottlingException" and attempt < max_attempts:
                    cls.log.warning(
                        f"ThrottlingException: Waiting for {sleep_times[attempt]} seconds before retrying..."
                    )
                    time.sleep(sleep_times[attempt])
                else:
                    if error_code == "ThrottlingException":
                        cls.log.critical(f"AWS Throttling Exception: Failed after {max_attempts} attempts!")
                    cls.log.critical(f"AWS Response: {error.response}")
                    break  # Exit the loop

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

        # Force Refresh should be used sparingly, it can cause AWS Throttling
        if force_refresh:
            msg = f"Getting {category} Metadata with force_refresh=True..."
            cls.log.warning(msg)
            cls.log.monitor(msg)

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

            # Submit the data refresh task to the thread pool
            cls.thread_pool.submit(cls.refresh_aws_data, category)

            # Return the stale data
            return cls.meta_cache.get(category)

        # If the metadata is fresh, just return it
        return cls.meta_cache.get(category)

    @classmethod
    def get_all_metadata(cls, force_refresh=False) -> dict:
        """Pull the metadata for ALL the Service Categories
        Args:
            force_refresh (bool, optional): Force a refresh of the metadata. Defaults to False.
        Returns:
            dict: The Metadata for ALL the Service Categories
        """

        # Force Refresh should be used sparingly, it can cause AWS Throttling
        if force_refresh:
            msg = "Getting ALL AWS Metadata with force_refresh=True..."
            cls.log.warning(msg)
            cls.log.monitor(msg)

        # Get the metadata for ALL the categories
        return {_category: cls.get_metadata(_category, force_refresh) for _category in ServiceCategory}

    @classmethod
    def shutdown(cls) -> None:
        """Wait for any open threads in our thread pool to finish"""
        cls.thread_pool.shutdown(wait=True)

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

    # Get S3 object sizes
    incoming_data_size = aws_broker.get_s3_object_sizes(ServiceCategory.INCOMING_DATA_S3)
    print(f"Incoming Data Size: {incoming_data_size} Bytes")

    """Memory Tests
    data_sources_size = aws_broker.get_s3_object_sizes(ServiceCategory.DATA_SOURCES_S3)
    print(f"Data Sources Size: {data_sources_size} Bytes")

    abalone_size = aws_broker.get_s3_object_sizes(ServiceCategory.DATA_SOURCES_S3, prefix="abalone_data")
    print(f"Abalone Size: {abalone_size} Bytes")
    """

    # Get the Metadata for ALL the categories (with a force refresh)
    pprint(aws_broker.get_all_metadata(force_refresh=True))

    # Wait for any open threads to finish
    aws_broker.shutdown()
