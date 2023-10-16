"""SageWorksCache is a wrapper around Cache and RedisCache that allows us to
use RedisCache if it's available, and fall back to Cache if it's not.
"""
from sageworks.utils.cache import Cache
from sageworks.utils.redis_cache import RedisCache


class SageWorksCache:
    # Initialize the actual cache at the class level
    _actual_cache = None

    @classmethod
    def _initialize_cache(cls, expire=None, prefix="", postfix=""):
        if RedisCache().check():
            return RedisCache(expire=expire, prefix=prefix, postfix=postfix)
        else:
            return Cache(expire=expire)

    def __init__(self, expire=None, prefix="", postfix=""):
        """SageWorksCache Initialization
        Args:
            expire: the number of seconds to keep items in the redis_cache
            prefix: the prefix to use for all keys
            postfix: the postfix to use for all keys
        """
        # Initialize the actual cache if it hasn't been initialized yet
        if SageWorksCache._actual_cache is None:
            SageWorksCache._actual_cache = self._initialize_cache(expire, prefix, postfix)

    def set(self, key, value):
        self._actual_cache.set(key, value)

    def get(self, key):
        return self._actual_cache.get(key)

    def delete(self, key):
        self._actual_cache.delete(key)

    def check(self):
        return self._actual_cache.check()

