"""CachedArtifactMixin for caching method results in Artifact subclasses"""

import logging
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from workbench.utils.workbench_cache import WorkbenchCache


class CachedArtifactMixin:
    """Mixin for caching methods in Artifact subclasses"""

    # Class-level caches, thread pool, and shutdown flag
    log = logging.getLogger("workbench")
    artifact_cache = WorkbenchCache(prefix="artifact_cache")
    fresh_cache = WorkbenchCache(prefix="artifact_fresh_cache", expire=10)
    thread_pool = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def _flatten_redis_key(method, *args, **kwargs):
        """Flatten the args and kwargs into a single string"""
        arg_str = "_".join(str(arg) for arg in args)
        kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return f"{method.__name__}_{arg_str}_{kwarg_str}".replace(" ", "").replace("'", "")

    @classmethod
    def cache_result(cls, method):
        """Decorator to cache method results"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Cache key includes the class name, instance UUID, and method args/kwargs
            class_name = self.__class__.__name__.lower()
            cache_key = f"{class_name}_{self.uuid}_{cls._flatten_redis_key(method, *args, **kwargs)}"

            # Get the cached value and check if a refresh is needed
            cached_value = cls.artifact_cache.get(cache_key)
            cache_fresh = cls.fresh_cache.get(cache_key)

            # Check for the blocking case (no cached value)
            if cached_value is None:
                self.log.important(f"Blocking: Invoking method {method.__name__} for {cache_key}")
                cls.fresh_cache.set(cache_key, True)
                result = method(self, *args, **kwargs)
                cls.artifact_cache.set(cache_key, result)
                return result

            # Stale cache: Refresh in the background if enabled and no refresh is already in progress
            if WorkbenchCache.refresh_enabled and cache_fresh is None:
                self.log.debug(f"Async: Refresh thread started: {cache_key}...")
                cls.fresh_cache.set(cache_key, True)
                cls.thread_pool.submit(cls._refresh_data_in_background, self, cache_key, method, *args, **kwargs)

            # Return the cached value (fresh or stale)
            return cached_value

        return wrapper

    @classmethod
    def _refresh_data_in_background(cls, instance, cache_key, method, *args, **kwargs):
        """Background data refresh method"""
        try:
            result = method(instance, *args, **kwargs)
            cls.artifact_cache.set(cache_key, result)
        except Exception as e:
            instance.log.error(f"Error refreshing data for {cache_key}: {e}")

    @classmethod
    def _shutdown(cls):
        """Explicitly shutdown the thread pool, if needed.
        Note: You should NOT call this method unless you know what you're doing."""
        if cls.thread_pool:
            cls.log.important("Shutting down the ThreadPoolExecutor...")
            time.sleep(10)
            try:
                cls.thread_pool.shutdown(wait=True)  # Gracefully shutdown
            except RuntimeError as e:
                cls.log.error(f"Error during thread pool shutdown: {e}")
            finally:
                cls.thread_pool = None


# Example usage of CachedModel
if __name__ == "__main__":
    from pprint import pprint
    from workbench.cached.cached_model import CachedModel

    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    CachedArtifactMixin._shutdown()
