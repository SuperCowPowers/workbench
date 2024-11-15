"""CachedArtifactMixin for caching method results in Artifact subclasses"""

import logging
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from sageworks.utils.sageworks_cache import SageWorksCache


class CachedArtifactMixin:
    """Mixin for caching methods in Artifact subclasses"""

    # Class-level caches, thread pool, and shutdown flag
    log = logging.getLogger("sageworks")
    artifact_cache = SageWorksCache(prefix="artifact_cache")
    fresh_cache = SageWorksCache(prefix="artifact_fresh_cache", expire=90)
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
            # Include the uuid in the cache key if available
            cache_key = f"{self.uuid}_{cls._flatten_redis_key(method, *args, **kwargs)}"
            if SageWorksCache.refresh_enabled and cls.fresh_cache.get(cache_key) is None:
                self.log.debug(f"Async: Results for {cache_key} refresh thread started...")
                cls.fresh_cache.set(cache_key, True)
                cls.thread_pool.submit(cls._refresh_data_in_background, self, cache_key, method, *args, **kwargs)

            cached_value = cls.artifact_cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            self.log.important(f"Blocking: Invoking method {method.__name__} for {cache_key}")
            result = method(self, *args, **kwargs)
            cls.artifact_cache.set(cache_key, result)
            return result

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
    from sageworks.cached.cached_model import CachedModel

    my_model = CachedModel("abalone-regression")
    pprint(my_model.summary())
    pprint(my_model.details())
    pprint(my_model.health_check())
    CachedArtifactMixin._shutdown()
