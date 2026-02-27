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
            result = cached_entry["_result"]

            # Guard against corrupted cache entries (e.g., failed DataFrame deserialization)
            if isinstance(result, dict) and "__dataframe__" in result:
                self.log.warning(f"Corrupted cache entry for {cache_key}, refetching...")
                self.meta_cache.delete(cache_key)
            elif (now - cached_entry.get("_cached_at", 0)) < self._cache_ttl:
                return result

        # Stale or first access: fetch fresh data
        result = method(self, *args, **kwargs)
        self.meta_cache.set(cache_key, {"_result": result, "_cached_at": now})

        # Update the Modified registry if this is a list method with Modified timestamps
        config = CachedMeta._registry_config.get(method.__name__)
        if (
            config is not None
            and isinstance(result, pd.DataFrame)
            and not result.empty
            and "Modified" in result.columns
        ):
            name_column = config["name_column"]
            new_entries = dict(zip(result[name_column], result["Modified"]))
            existing = self.modified_registry.get(method.__name__) or {}

            # Merge: add new, remove deleted, keep max(existing, new) for updates
            merged = {}
            for name, ts in new_entries.items():
                existing_ts = existing.get(name)
                merged[name] = max(existing_ts, ts) if existing_ts else ts
            self.modified_registry.set(method.__name__, merged)

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

    # Artifact registry configuration
    # This is the central lookup that maps artifact types to their registry keys,
    # the DataFrame column containing artifact names, and the base classes used to
    # identify artifact objects. CachedMeta needs this to:
    #   - Track Modified timestamps per artifact (keyed by registry key)
    #   - Know which column holds the artifact name in list DataFrames
    #   - Resolve any artifact object (Model, CachedModel, etc.) to its registry key
    #   - Call per-artifact detail methods for incremental refresh
    _registry_config = {
        "data_sources": {"name_column": "Name", "base_class": "AthenaSource"},
        "feature_sets": {"name_column": "Feature Group", "base_class": "FeatureSetCore",
                         "detail_method": "_feature_set_detail_row"},
        "models": {"name_column": "Model Group", "base_class": "ModelCore",
                   "detail_method": "_model_detail_row"},
        "endpoints": {"name_column": "Name", "base_class": "EndpointCore",
                      "detail_method": "_endpoint_detail_row"},
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
        if not details:
            return super().feature_sets(details=False)
        return self._refresh_details("feature_sets")

    @cache_result
    def models(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        if not details:
            return super().models(details=False)
        return self._refresh_details("models")

    @cache_result
    def endpoints(self, details: bool = False) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Args:
            details (bool, optional): Include detailed information. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        if not details:
            return super().endpoints(details=False)
        return self._refresh_details("endpoints")

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

    def _resolve_registry_key(self, artifact) -> str:
        """Resolve any artifact object to its registry key by inspecting the class hierarchy.

        Works with any artifact type: Model, CachedModel, ModelCore, Endpoint, etc.

        Args:
            artifact: Any Workbench artifact object.

        Returns:
            str: The registry key (e.g., "models", "endpoints"), or None if not recognized.
        """
        class_names = {cls.__name__ for cls in type(artifact).__mro__}
        for key, config in self._registry_config.items():
            if config["base_class"] in class_names:
                return key
        return None

    def get_modified_timestamp(self, artifact):
        """Look up an artifact's Modified timestamp in the registry.

        Args:
            artifact: Any Workbench artifact object (Model, CachedModel, Endpoint, etc.)

        Returns:
            datetime: The Modified timestamp, or None if not found
        """
        registry_key = self._resolve_registry_key(artifact)
        if registry_key is None:
            return None
        entries = self.modified_registry.get(registry_key) or {}
        return entries.get(artifact.name)

    def update_modified_timestamp(self, artifact):
        """Update an artifact's Modified timestamp to now.

        Pokes the registry so the artifact is detected as stale on next access.

        Args:
            artifact: Any Workbench artifact object (Model, CachedModel, Endpoint, etc.)
        """
        registry_key = self._resolve_registry_key(artifact)
        if registry_key is None:
            raise ValueError(f"Cannot determine registry key for {type(artifact).__name__}")
        now = datetime.now(timezone.utc)
        entries = self.modified_registry.get(registry_key) or {}
        entries[artifact.name] = now
        self.modified_registry.set(registry_key, entries)

    def _refresh_details(self, list_method: str) -> pd.DataFrame:
        """Incremental detail refresh using the modified registry.

        Compares cached detail rows against the registry, refetches only stale artifacts.

        Args:
            list_method (str): The list method name ("feature_sets", "models", or "endpoints")

        Returns:
            pd.DataFrame: The refreshed details DataFrame
        """
        config = self._registry_config[list_method]
        name_col = config["name_column"]
        detail_method = getattr(self, config["detail_method"])

        # Step 1: Get fresh lightweight list (also updates registry via decorator)
        lightweight_df = getattr(self, list_method)(details=False)
        if lightweight_df.empty:
            return lightweight_df

        # Step 2: Get registry and cached details
        registry = self.modified_registry.get(list_method) or {}
        cached_df = self._get_previous_result(list_method, details=True)

        # Step 3: If no cached details, fetch everything
        if cached_df is None or not isinstance(cached_df, pd.DataFrame) or cached_df.empty:
            rows = [detail_method(name) for name in lightweight_df[name_col]]
            df = pd.DataFrame(rows).convert_dtypes()
            if not df.empty:
                df.sort_values(by="Created", ascending=False, inplace=True)
            return df

        # Step 4: Identify stale artifacts (new or cached timestamp < registry timestamp)
        current_names = set(lightweight_df[name_col])
        cached_names = set(cached_df[name_col])
        stale_names = set()

        for name in current_names:
            if name not in cached_names:
                stale_names.add(name)  # New artifact
                continue
            registry_ts = registry.get(name)
            if registry_ts is not None:
                cached_modified = cached_df.loc[cached_df[name_col] == name, "Modified"].iloc[0]
                if cached_modified < registry_ts:
                    stale_names.add(name)  # Registry says it changed

        # Step 5: Refetch stale artifacts and merge into cached DataFrame
        if stale_names:
            self.log.info(f"{list_method} details: {len(current_names) - len(stale_names)} reused, {len(stale_names)} refreshed")
            fresh_rows = [detail_method(name) for name in stale_names]
            fresh_df = pd.DataFrame(fresh_rows)

            # Stamp fresh rows with the registry timestamp so the artifact cache
            # (CachedArtifactMixin) and this registry agree after one refresh cycle.
            # Without this, the registry would keep the poke timestamp while the
            # fresh row has the AWS timestamp → stale again on the next check.
            for idx, row in fresh_df.iterrows():
                registry_ts = registry.get(row[name_col])
                if registry_ts is not None:
                    fresh_df.at[idx, "Modified"] = registry_ts

            # Remove old rows for stale artifacts and append fresh ones
            cached_df = cached_df[~cached_df[name_col].isin(stale_names)]
            cached_df = pd.concat([cached_df, fresh_df], ignore_index=True)

        # Step 6: Remove deleted artifacts (in cached but not in lightweight list)
        cached_df = cached_df[cached_df[name_col].isin(current_names)].copy()

        if not cached_df.empty:
            cached_df.sort_values(by="Created", ascending=False, inplace=True)
            cached_df = cached_df.convert_dtypes()

        return cached_df

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
