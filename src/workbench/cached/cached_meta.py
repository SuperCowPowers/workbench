"""CachedMeta: A class that provides caching for the Meta() class"""

import logging
import time
import pandas as pd
from datetime import datetime, timezone
from functools import wraps

# Workbench Imports
from workbench.core.cloud_platform.cloud_meta import CloudMeta
from workbench.utils.workbench_cache import WorkbenchCache


def cache_result(method):
    """Decorator to cache method results with a TTL-based staleness check"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cache_key = WorkbenchCache.flatten_key(method, *args, **kwargs)
        now = time.time()

        # Check if we have a cached result that's still fresh
        cached_entry = self.meta_cache.get(cache_key)
        if cached_entry is not None and isinstance(cached_entry, dict) and "_result" in cached_entry:
            if (now - cached_entry.get("_cached_at", 0)) < self._cache_ttl:
                return cached_entry["_result"]

        # Stale or first access: fetch fresh data
        result = method(self, *args, **kwargs)
        self.meta_cache.set(cache_key, {"_result": result, "_cached_at": now})

        # Update the Modified registry if this is a list method with Modified timestamps
        name_column = CachedMeta._registry_config.get(method.__name__)
        if (
            name_column is not None
            and isinstance(result, pd.DataFrame)
            and not result.empty
            and "Modified" in result.columns
        ):
            self.modified_registry.set(method.__name__, dict(zip(result[name_column], result["Modified"])))

        return result

    return wrapper


class CachedMeta(CloudMeta):
    """CachedMeta: Singleton class for caching list-level metadata.

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
       meta.feature_sets()
       meta.models()
       meta.endpoints()
       meta.views()
       ```
    """

    _instance = None  # Class attribute to hold the singleton instance
    _cache_ttl = 30  # 30 seconds

    # Registry config maps list method names to the DataFrame column containing artifact names
    _registry_config = {
        "data_sources": "Name",
        "feature_sets": "Feature Group",
        "models": "Model Group",
        "endpoints": "Name",
    }

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

        # Meta Cache for list method results
        self.meta_cache = WorkbenchCache(prefix="meta")

        # Modified timestamp registry (Redis-backed for cross-process sharing)
        self.modified_registry = WorkbenchCache(prefix="modified_registry")

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
        previous_df = self._get_previous_result("feature_sets", details=True) if details else None
        return super().feature_sets(details=details, previous_df=previous_df)

    @cache_result
    def models(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        previous_df = self._get_previous_result("models", details=True) if details else None
        return super().models(details=details, previous_df=previous_df)

    @cache_result
    def endpoints(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        previous_df = self._get_previous_result("endpoints", details=True) if details else None
        return super().endpoints(details=details, previous_df=previous_df)

    def get_modified_registry(self, list_method: str = None) -> dict:
        """Get the Modified timestamp registry

        Args:
            list_method (str, optional): Filter to a specific list method (e.g., "models"). Defaults to None (all).

        Returns:
            dict: The full registry or a single list method's entries
        """
        if list_method:
            return self.modified_registry.get(list_method) or {}
        registry = {}
        for method_name in self._registry_config:
            entries = self.modified_registry.get(method_name)
            if entries:
                registry[method_name] = entries
        return registry

    def get_modified_timestamp(self, artifact):
        """Look up a Cached Artifact's Modified timestamp

        Args:
            artifact (CachedArtifact): A Cached Artifact object (CachedModel, CachedEndpoint, etc.)

        Returns:
            datetime: The Modified timestamp, or None if not found
        """
        list_method = artifact._list_method
        entries = self.modified_registry.get(list_method) or {}
        return entries.get(artifact.name)

    def update_modified_timestamp(self, artifact):
        """Update a Cached Artifact's Modified timestamp to now.

        This pokes both the registry (for CachedArtifactMixin staleness) and the
        cached previous_df (for incremental detail reuse), so the entire caching
        system sees the artifact as dirty and refetches on next access.

        Args:
            artifact (CachedArtifact): A Cached Artifact object (CachedModel, CachedEndpoint, etc.)
        """
        list_method = artifact._list_method
        now = datetime.now(timezone.utc)

        # Update the registry in Redis
        entries = self.modified_registry.get(list_method) or {}
        entries[artifact.name] = now
        self.modified_registry.set(list_method, entries)

        # Update the cached previous_df so the previous_df reuse path also sees the change
        cache_key = WorkbenchCache.flatten_key(list_method, details=True)
        cached_entry = self.meta_cache.get(cache_key)
        if cached_entry is not None and isinstance(cached_entry, dict) and "_result" in cached_entry:
            df = cached_entry["_result"]
            name_column = self._registry_config.get(list_method)
            if name_column and isinstance(df, pd.DataFrame) and not df.empty:
                mask = df[name_column] == artifact.name
                if mask.any():
                    df.loc[mask, "Modified"] = now
                    self.meta_cache.set(cache_key, cached_entry)

    def _get_previous_result(self, method_name, **kwargs):
        """Get the previously cached result for incremental detail updates"""
        cache_key = WorkbenchCache.flatten_key(method_name, **kwargs)
        cached_entry = self.meta_cache.get(cache_key)
        if cached_entry is not None and isinstance(cached_entry, dict) and "_result" in cached_entry:
            return cached_entry["_result"]
        return None

    def __repr__(self):
        return f"CachedMeta()\n\t{repr(self.meta_cache)}\n\t{super().__repr__()}"


if __name__ == "__main__":
    """Exercise the Workbench CachedMeta Class"""

    # Pandas Display Options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Create the class
    meta = CachedMeta()
    print(f"Check: {meta.check()}")
    print(meta)

    # List methods
    print("\n\n*** Data Sources ***")
    print(meta.data_sources())

    print("\n\n*** Feature Sets ***")
    print(meta.feature_sets())

    print("\n\n*** Models ***")
    print(meta.models())

    print("\n\n*** Endpoints ***")
    print(meta.endpoints())

    # Check the Modified registry
    print("\n\n*** Modified Registry ***")
    for key, value in meta.get_modified_registry().items():
        print(f"\n{key}: {len(value)} artifacts")

    # Second call to demonstrate caching
    print("\n\n*** Data Sources (cached) ***")
    print(meta.data_sources())
