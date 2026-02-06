"""CachedArtifactMixin for caching method results in Artifact subclasses"""

import logging
import threading
import time
from contextlib import contextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from workbench.utils.workbench_cache import WorkbenchCache


class CachedArtifactMixin:
    """Mixin for caching methods in Artifact subclasses"""

    # Class-level caches, thread pool, and shutdown flag
    log = logging.getLogger("workbench")
    artifact_cache = WorkbenchCache(prefix="artifact_cache")
    fresh_cache = WorkbenchCache(prefix="artifact_fresh_cache", expire=120)
    thread_pool = ThreadPoolExecutor(max_workers=5)

    # Thread-local storage for the skip_refresh flag
    _local = threading.local()

    @classmethod
    @contextmanager
    def no_refresh(cls):
        """Context manager to skip background refresh threads.

        Useful for batch operations where you want fast cached reads
        without spawning background refresh threads.

        Example:
            with CachedArtifactMixin.no_refresh():
                for name in model_names:
                    model = CachedModel(name)
                    # ... do work ...
        """
        cls._local.skip_refresh = True
        try:
            yield
        finally:
            cls._local.skip_refresh = False

    @classmethod
    def _should_refresh(cls) -> bool:
        """Check if we should launch background refresh threads."""
        return not getattr(cls._local, "skip_refresh", False)

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
            # Cache key includes the class name, instance Name, and method args/kwargs
            class_name = self.__class__.__name__.lower()
            cache_key = f"{class_name}_{self.name}_{cls._flatten_redis_key(method, *args, **kwargs)}"

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

            # Stale cache: Refresh in the background (unless skip_refresh is set)
            if cache_fresh is None and cls._should_refresh():
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
    # Second call to demonstrate caching
    pprint(my_model.summary())
    pprint(my_model.details())
    CachedArtifactMixin._shutdown()
