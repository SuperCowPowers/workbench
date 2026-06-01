"""CachedArtifactMixin: Caching for Artifact subclasses using Modified timestamps for stale detection"""

import logging
from functools import wraps

from workbench.utils.workbench_cache import WorkbenchCache


class CachedArtifactMixin:
    """Mixin for caching methods in Artifact subclasses.

    Uses Modified timestamps from CachedMeta's registry to detect staleness.
    Fresh artifacts return instantly from cache. Stale artifacts block and refetch.
    """

    # Class-level cache for artifact method results
    log = logging.getLogger("workbench")
    artifact_cache = WorkbenchCache(prefix="artifact_cache")
    max_cached_result_bytes = 1024 * 1024

    @classmethod
    def _result_size_bytes(cls, result):
        """Best-effort size estimate for deciding whether a result belongs in Redis."""
        try:
            if hasattr(result, "memory_usage"):
                memory_usage = result.memory_usage(deep=True)
                if hasattr(memory_usage, "sum"):
                    return int(memory_usage.sum())
                return int(memory_usage)
            return len(str(result))
        except Exception as e:
            cls.log.warning(f"Cache: Could not estimate result size: {e}")
            return None

    @classmethod
    def cache_result(cls, method):
        """Decorator to cache method results using Modified timestamps for stale detection"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Build cache key (class name + instance name + method + args)
            class_name = self.__class__.__name__.lower()
            cache_key = f"{class_name}_{self.name}_{WorkbenchCache.flatten_key(method, *args, **kwargs)}"

            # Get the cached entry (stored as {"_result": ..., "_modified": ...})
            cached_entry = cls.artifact_cache.get(cache_key)

            # Lazy import to avoid circular dependency (CachedMeta ↔ CachedArtifactMixin)
            from workbench.cached.cached_meta import CachedMeta

            current_modified = CachedMeta().get_modified_timestamp(self)

            # Check if cached value is fresh (cached timestamp >= registry timestamp)
            if cached_entry is not None and isinstance(cached_entry, dict) and "_result" in cached_entry:
                cached_modified = cached_entry.get("_modified")
                if current_modified is not None and cached_modified is not None and cached_modified >= current_modified:
                    return cached_entry["_result"]
                else:
                    self.log.info(f"Stale: Refreshing {method.__name__} for {self.name}")

            # Stale or first access: fetch fresh data
            result = method(self, *args, **kwargs)
            result_size = cls._result_size_bytes(result)
            if result_size is not None and result_size > cls.max_cached_result_bytes:
                self.log.warning(
                    "Cache: Skipping oversized cached result "
                    f"({cache_key}: {result_size} bytes > {cls.max_cached_result_bytes})"
                )
                return result

            try:
                cls.artifact_cache.set(cache_key, {"_result": result, "_modified": current_modified})
            except Exception as e:
                self.log.warning(f"Cache: Skipping cached result for {cache_key} after cache write failed: {e}")
            return result

        return wrapper

    def refresh(self) -> int:
        """Invalidate all cached method results for this artifact.

        Forces subsequent method calls to bypass the cache and re-fetch fresh
        data. Useful when post-processing logic has changed but the underlying
        artifact's Modified timestamp has not.

        Returns:
            int: Number of cache entries deleted.
        """
        class_name = self.__class__.__name__.lower()
        prefix = f"{class_name}_{self.name}_"
        keys = self.artifact_cache.list_subkeys(prefix)
        for key in keys:
            self.artifact_cache.delete(key)
        self.log.info(f"Refresh: Cleared {len(keys)} cached entries for {self.name}")
        return len(keys)


# Example usage of CachedModel
if __name__ == "__main__":
    from pprint import pprint

    from workbench.cached.cached_model import CachedModel

    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    # Second call to demonstrate caching
    pprint(my_model.summary())
    pprint(my_model.details())
