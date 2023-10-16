"""SageWorksCache is a wrapper around Cache and RedisCache that allows us to
use RedisCache if it's available, and fall back to Cache if it's not.
"""
from sageworks.utils.cache import Cache
from sageworks.utils.redis_cache import RedisCache


class SageWorksCache:
    def __init__(self, expire=None, prefix="", postfix=""):
        """SageWorksCache Initialization
        Args:
            expire: the number of seconds to keep items in the redis_cache
            prefix: the prefix to use for all keys
            postfix: the postfix to use for all keys
        """
        if RedisCache().check():
            self._actual_cache = RedisCache(expire=expire, prefix=prefix, postfix=postfix)
        else:
            # If Redis isn't available, fall back to Cache
            # Note: Since we have a 'separate' Cache, we don't need prefix/postfix logic
            self._actual_cache = Cache(expire=expire)

    def set(self, key, value):
        self._actual_cache.set(key, value)

    def get(self, key):
        return self._actual_cache.get(key)

    def delete(self, key):
        self._actual_cache.delete(key)

    def check(self):
        return self._actual_cache.check()
