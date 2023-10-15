"""SageWorksCache is a wrapper around Cache and RedisCache that allows us to
use RedisCache if it's available, and fall back to Cache if it's not.
"""
from sageworks.utils.cache import Cache
from sageworks.utils.redis_cache import RedisCache


class SageWorksCache:
    def __init__(self, expire=None, postfix=""):
        if RedisCache().check():
            self._actual_cache = RedisCache(expire=expire, postfix=postfix)
        else:
            self._actual_cache = Cache(expire=expire)

    def set(self, key, value):
        self._actual_cache.set(key, value)

    def get(self, key):
        return self._actual_cache.get(key)

    def check(self):
        return self._actual_cache.check()
