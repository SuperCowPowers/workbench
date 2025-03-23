"""CachedMeta: A class that provides caching for the Meta() class"""

import logging
from typing import Union
import pandas as pd
from functools import wraps
from concurrent.futures import ThreadPoolExecutor


# Workbench Imports
from workbench.core.cloud_platform.cloud_meta import CloudMeta
from workbench.utils.workbench_cache import WorkbenchCache


# Decorator to cache method results from the Meta class
# Note: This has to be outside the class definition to work properly in Python 3.9
#       When we deprecated support for 3.9, move this back into the class definition
def cache_result(method):
    """Decorator to cache method results in meta_cache"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Create a unique cache key based on the method name and arguments
        cache_key = CachedMeta._flatten_redis_key(method, *args, **kwargs)

        # Check for fresh data, spawn thread to refresh if stale
        if WorkbenchCache.refresh_enabled and self.fresh_cache.get(cache_key) is None:
            self.log.debug(f"Async: Metadata for {cache_key} refresh thread started...")
            self.fresh_cache.set(cache_key, True)  # Mark as refreshed

            # Spawn a thread to refresh data without blocking
            self.thread_pool.submit(self._refresh_data_in_background, cache_key, method, *args, **kwargs)

        # Return data (fresh or stale) if available
        cached_value = self.meta_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Fall back to calling the method if no cached data found
        self.log.important(f"Blocking: Getting Metadata for {cache_key}")
        result = method(self, *args, **kwargs)
        self.meta_cache.set(cache_key, result)
        return result

    return wrapper


class CachedMeta(CloudMeta):
    """CachedMeta: Singleton class for caching metadata functionality.

    Common Usage:
       ```python
       from workbench.cached.cached_meta import CachedMeta
       meta = CachedMeta()

       # Get the AWS Account Info
       meta.account()
       meta.config()

       # These are 'list' methods
       meta.etl_jobs()
       meta.data_sources()
       meta.feature_sets(details=True/False)
       meta.models(details=True/False)
       meta.endpoints()
       meta.views()

       # These are 'describe' methods
       meta.data_source("abalone_data")
       meta.feature_set("abalone_features")
       meta.model("abalone-regression")
       meta.endpoint("abalone-endpoint")
       ```
    """

    _instance = None  # Class attribute to hold the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CachedMeta, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """CachedMeta Initialization"""
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent reinitialization

        self.log = logging.getLogger("workbench")
        self.log.important("Initializing CachedMeta...")
        super().__init__()

        # Create both our Meta Cache and Fresh Cache (tracks if data is stale)
        self.meta_cache = WorkbenchCache(prefix="meta")
        self.fresh_cache = WorkbenchCache(prefix="meta_fresh", expire=90)  # 90-second expiration

        # Create a ThreadPoolExecutor for refreshing stale data
        self.thread_pool = ThreadPoolExecutor(max_workers=5)

        # Mark the instance as initialized
        self._initialized = True

    def check(self):
        """Check if our underlying caches are working"""
        return self.meta_cache.check()

    def list_meta_cache(self):
        """List the current Meta Cache"""
        return self.meta_cache.list_keys()

    def clear_meta_cache(self):
        """Clear the current Meta Cache"""
        self.meta_cache.clear()

    @cache_result
    def account(self) -> dict:
        """Cloud Platform Account Info

        Returns:
            dict: Cloud Platform Account Info
        """
        return super().account()

    @cache_result
    def config(self) -> dict:
        """Return the current Workbench Configuration

        Returns:
            dict: The current Workbench Configuration
        """
        return super().config()

    @cache_result
    def incoming_data(self) -> pd.DataFrame:
        """Get summary data about data in the incoming raw data

        Returns:
            pd.DataFrame: A summary of the incoming raw data
        """
        return super().incoming_data()

    @cache_result
    def etl_jobs(self) -> pd.DataFrame:
        """Get summary data about Extract, Transform, Load (ETL) Jobs

        Returns:
            pd.DataFrame: A summary of the ETL Jobs deployed in the Cloud Platform
        """
        return super().etl_jobs()

    @cache_result
    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Data Sources deployed in the Cloud Platform
        """
        return super().data_sources()

    @cache_result
    def views(self, database: str = "workbench") -> pd.DataFrame:
        """Get a summary of the all the Views, for the given database, in AWS

        Args:
            database (str, optional): Glue database. Defaults to 'workbench'.

        Returns:
            pd.DataFrame: A summary of all the Views, for the given database, in AWS
        """
        return super().views(database=database)

    @cache_result
    def feature_sets(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Feature Sets deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Feature Sets deployed in the Cloud Platform
        """
        return super().feature_sets(details=details)

    @cache_result
    def models(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        return super().models(details=details)

    @cache_result
    def endpoints(self) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        return super().endpoints()

    @cache_result
    def glue_job(self, job_name: str) -> Union[dict, None]:
        """Get the details of a specific Glue Job

        Args:
            job_name (str): The name of the Glue Job

        Returns:
            dict: The details of the Glue Job (None if not found)
        """
        return super().glue_job(job_name=job_name)

    @cache_result
    def data_source(self, data_source_name: str, database: str = "workbench") -> Union[dict, None]:
        """Get the details of a specific Data Source

        Args:
            data_source_name (str): The name of the Data Source
            database (str, optional): The Glue database. Defaults to 'workbench'.

        Returns:
            dict: The details of the Data Source (None if not found)
        """
        return super().data_source(data_source_name=data_source_name, database=database)

    @cache_result
    def feature_set(self, feature_set_name: str) -> Union[dict, None]:
        """Get the details of a specific Feature Set

        Args:
            feature_set_name (str): The name of the Feature Set

        Returns:
            dict: The details of the Feature Set (None if not found)
        """
        return super().feature_set(feature_set_name=feature_set_name)

    @cache_result
    def model(self, model_name: str) -> Union[dict, None]:
        """Get the details of a specific Model

        Args:
            model_name (str): The name of the Model

        Returns:
            dict: The details of the Model (None if not found)
        """
        return super().model(model_name=model_name)

    @cache_result
    def endpoint(self, endpoint_name: str) -> Union[dict, None]:
        """Get the details of a specific Endpoint

        Args:
            endpoint_name (str): The name of the Endpoint

        Returns:
            dict: The details of the Endpoint (None if not found)
        """
        return super().endpoint(endpoint_name=endpoint_name)

    def _refresh_data_in_background(self, cache_key, method, *args, **kwargs):
        """Background task to refresh AWS metadata."""
        result = method(self, *args, **kwargs)
        self.meta_cache.set(cache_key, result)
        self.log.debug(f"Updated Metadata for {cache_key}")

    @staticmethod
    def _flatten_redis_key(method, *args, **kwargs):
        """Flatten the args and kwargs into a single string"""
        arg_str = "_".join(str(arg) for arg in args)
        kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return f"{method.__name__}_{arg_str}_{kwarg_str}".replace(" ", "").replace("'", "")

    def __del__(self):
        """Destructor to shut down the thread pool gracefully."""
        self.close()

    def close(self):
        """Explicitly close the thread pool, if needed."""
        if self.thread_pool:
            self.log.important("Shutting down the ThreadPoolExecutor...")
            try:
                self.thread_pool.shutdown(wait=True)  # Gracefully shutdown
            except RuntimeError as e:
                self.log.error(f"Error during thread pool shutdown: {e}")
            finally:
                self.thread_pool = None

    def __repr__(self):
        return f"CachedMeta()\n\t{repr(self.meta_cache)}\n\t{super().__repr__()}"


if __name__ == "__main__":
    """Exercise the Workbench AWSCachedMeta Class"""
    from pprint import pprint
    import time

    # Pandas Display Options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create the class
    meta = CachedMeta()
    print(f"Check: {meta.check()}")

    # Test the __repr__ method
    print(meta)

    # List the current Meta Cache
    meta.list_meta_cache()

    # Clear the current Meta Cache
    # meta.clear_meta_cache()

    # Get the AWS Account Info
    print("*** AWS Account ***")
    pprint(meta.account())

    # Get the Workbench Configuration
    print("*** Workbench Configuration ***")
    pprint(meta.config())

    # Get the Incoming Data
    print("\n\n*** Incoming Data ***")
    print(meta.incoming_data())

    # Get the AWS Glue Jobs (ETL Jobs)
    print("\n\n*** ETL Jobs ***")
    print(meta.etl_jobs())

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    print(meta.data_sources())

    # Get the Views (Data Sources)
    print("\n\n*** Views (Data Sources) ***")
    print(meta.views("workbench"))

    # Get the Views (Feature Sets)
    print("\n\n*** Views (Feature Sets) ***")
    fs_views = meta.views("sagemaker_featurestore")
    print(fs_views)

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets())

    # Get the Models
    print("\n\n*** Models ***")
    start_time = time.time()
    pprint(meta.models())
    print(f"Elapsed Time Model (no details): {time.time() - start_time:.2f}")

    # Get the Models with Details
    print("\n\n*** Models with Details ***")
    start_time = time.time()
    pprint(meta.models(details=True))
    print(f"Elapsed Time Model (with details): {time.time() - start_time:.2f}")

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())

    # Test out the specific artifact details methods
    print("\n\n*** Glue Job Details ***")
    pprint(meta.glue_job("Glue_Job_1"))
    print("\n\n*** DataSource Details ***")
    pprint(meta.data_source("abalone_data"))
    print("\n\n*** FeatureSet Details ***")
    pprint(meta.feature_set("abalone_features"))
    print("\n\n*** Model Details ***")
    pprint(meta.model("abalone-regression"))
    print("\n\n*** Endpoint Details ***")
    pprint(meta.endpoint("abalone-regression"))

    # Test out a non-existent model
    print("\n\n*** Model Doesn't Exist ***")
    pprint(meta.model("non-existent-model"))
