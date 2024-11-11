import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from sageworks.utils.sageworks_cache import SageWorksCache


# Mixin class for caching functionality
class CachedArtifactMixin:
    """Mixin for caching methods in Artifact subclasses"""

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "log"):
            self.log = logging.getLogger("sageworks")
        super().__init__(*args, **kwargs)
        self.artifact_cache = SageWorksCache(prefix=f"{self.__class__.__name__.lower()}:{self.uuid}")
        self.fresh_cache = SageWorksCache(prefix=f"{self.__class__.__name__.lower()}_fresh:{self.uuid}", expire=30)
        self.thread_pool = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def cache_result(method):
        """Decorator to cache method results"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            cache_key = CachedArtifactMixin._flatten_redis_key(method, *args, **kwargs)

            if self.fresh_cache.get(cache_key) is None:
                self.log.important(f"Async: Results for {cache_key} refresh thread started...")
                self.fresh_cache.set(cache_key, True)
                self.thread_pool.submit(self._refresh_data_in_background, cache_key, method, *args, **kwargs)

            cached_value = self.artifact_cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            self.log.important(f"Blocking: Invoking method {method.__name__} for {cache_key}")
            result = method(self, *args, **kwargs)
            self.artifact_cache.set(cache_key, result)
            return result

        return wrapper

    @staticmethod
    def _flatten_redis_key(method, *args, **kwargs):
        """Flatten the args and kwargs into a single string"""
        arg_str = "_".join(str(arg) for arg in args)
        kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return f"{method.__name__}_{arg_str}_{kwarg_str}".replace(" ", "").replace("'", "")

    def _refresh_data_in_background(self, cache_key, method, *args, **kwargs):
        """Background data refresh method"""
        try:
            result = method(self, *args, **kwargs)
            self.artifact_cache.set(cache_key, result)
        except Exception as e:
            self.log.error(f"Error refreshing data for {cache_key}: {e}")

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


if __name__ == "__main__":
    """Exercise the SageWorks CachingArtifactMixin Class"""
    from pprint import pprint
    from sageworks.api import Model

    class CachedModel(CachedArtifactMixin, Model):
        """Model class with caching functionality"""

        def __init__(self, uuid: str):
            super().__init__(uuid)

        @CachedArtifactMixin.cache_result
        def details(self):
            """Example method that will use caching"""
            return super().details()

    # Create a CachedModel instance
    my_model = CachedModel("abalone-regression")
    print("Model Details:")
    pprint(my_model.details())
