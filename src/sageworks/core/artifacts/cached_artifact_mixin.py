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
        self.fresh_cache = SageWorksCache(prefix=f"{self.__class__.__name__.lower()}:{self.uuid}_fresh", expire=30)
        self.thread_pool = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def cache_result(method):
        """Decorator to cache method results"""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            cache_key = f"{method.__name__}_{args}_{kwargs}"

            if self.fresh_cache.get(cache_key) is None:
                self.log.debug(f"Async: Results for {cache_key} refresh thread started...")
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

    def _refresh_data_in_background(self, cache_key, method, *args, **kwargs):
        """Background data refresh method"""
        try:
            result = method(self, *args, **kwargs)
            self.artifact_cache.set(cache_key, result)
        except Exception as e:
            self.log.error(f"Error refreshing data for {cache_key}: {e}")


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
            # Original implementation of the method
            return super().details()


    # Create a CachedModel instance
    my_model = CachedModel("abalone-regression")
    print("Model Details:")
    pprint(my_model.details())
