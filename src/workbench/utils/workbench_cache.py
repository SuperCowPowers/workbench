"""WorkbenchCache is a wrapper around Cache and RedisCache that allows us to
use RedisCache if it's available, and fall back to Cache if it's not.
"""

from pprint import pformat
from contextlib import contextmanager
from workbench.utils.cache import Cache
from workbench.utils.redis_cache import RedisCache

import logging

log = logging.getLogger("workbench")


# Context manager for disabling refresh
@contextmanager
def disable_refresh():
    log.warning("WorkbenchCache: Disabling Refresh")
    WorkbenchCache.refresh_enabled = False
    yield
    log.warning("WorkbenchCache: Enabling Refresh")
    WorkbenchCache.refresh_enabled = True


class WorkbenchCache:

    # Class attribute to control refresh treads (on/off)
    refresh_enabled = True

    def __init__(self, expire=None, prefix="", postfix=""):
        """WorkbenchCache Initialization
        Args:
            expire: the number of seconds to keep items in the cache
            prefix: the prefix to use for all keys
            postfix: the postfix to use for all keys
        """
        self.medium_data = 100 * 1024  # 100KB
        self.large_data = 1024 * 1024  # 1MB

        # Create a RedisCache instance and check its connection
        self._using_redis = True
        redis_cache = RedisCache(expire=expire, prefix=prefix, postfix=postfix)
        if redis_cache.check():
            self._actual_cache = redis_cache
        else:
            # If Redis isn't available, fall back to an In-Memory Cache
            self._using_redis = False
            log.error("Redis connect failed, using In-Memory Cache...")
            self._actual_cache = Cache(expire=expire, prefix=prefix, postfix=postfix)

    def set(self, key, value):
        # Check the size of the value
        size = len(str(value))
        if size > self.large_data:
            log.warning(f"Cache: Setting large value: ({key}: {size})")
        elif size > self.medium_data:
            log.info(f"Cache: Setting medium cache value: ({key}: {size})")
        self._actual_cache.set(key, value)

    def get(self, key):
        # Check the size of the value
        value = self._actual_cache.get(key)
        size = len(str(value))
        if size > self.large_data:
            log.info(f"Cache: Getting large value: ({key}: {size})")
        elif size > self.medium_data:
            log.debug(f"Cache: Getting medium value: ({key}: {size})")
        return value

    def delete(self, key):
        self._actual_cache.delete(key)

    def list_keys(self):
        return self._actual_cache.list_keys()

    def list_subkeys(self, key):
        return self._actual_cache.list_subkeys(key)

    def check(self):
        return self._using_redis

    def clear(self):
        return self._actual_cache.clear()

    def show_size_details(self, value):
        """Print the size of the sub-parts of the value"""
        try:
            size_details = self._size_details(value)
            log.warning("Cache: Large Data Details")
            formatted_details = pformat(size_details, width=40)  # Adjust width as needed
            for line in formatted_details.splitlines():
                log.warning(f"{line}")
        except Exception as e:
            log.error(f"Cache: Error getting size details {e}")

    def _size_details(self, value):
        """Return the size of the sub-parts of the value"""
        if isinstance(value, dict):
            return {k: self._size_details(v) for k, v in value.items()}
        elif isinstance(value, list) and value:
            return len(value) * self._size_details(str(value[0]))
        else:
            return len(str(value))

    def __repr__(self):
        return f"WorkbenchCache({repr(self._actual_cache)})"


if __name__ == "__main__":
    """Exercise the Workbench Cache class"""

    # Create the Workbench Cache
    my_cache = WorkbenchCache(prefix="test")

    # Test the __repr__ method
    print(my_cache)

    # Delete anything in the test database
    my_cache.clear()

    # Test storage
    my_cache.set("foo", "bar")
    assert my_cache.get("foo") == "bar"
    for key in my_cache.list_keys():
        print(key)
    my_cache.clear()

    # Test prefix/postfix
    my_cache = WorkbenchCache(prefix="pre", postfix="post")
    my_cache.set("foo", "bar")
    assert my_cache.get("foo") == "bar"
    my_cache.list_keys()
    for key in my_cache.list_keys():
        print(key)
    my_cache.clear()

    # Test medium and large data
    my_cache = WorkbenchCache(prefix="test")
    my_cache.set("foo", "a" * 1024)
    my_cache.set("bar", "a" * 1024 * 101)
    my_cache.set("baz", "a" * 1024 * 1025)
    my_cache.get("foo")
    my_cache.get("bar")
    my_cache.get("baz")
    my_cache.clear()

    # Test more complicated large data
    mega = 1024 * 1024
    large_data = {
        "a": ["foo"] * mega,
        "b": {
            "c": "bar" * mega,
            "d": {"e": [{"f": ["baz"] * mega}, {"g": ["z" * mega]}], "h": "bleh", "i": "blarg" * mega},
            "j": "blargblarg",
        },
    }
    my_cache.set("large", large_data)
    my_cache.get("large")
    my_cache.clear()

    # Test Dataframe with a named index
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.index.name = "named_index"
    print(df)
    my_cache.set("df", df)
    df = my_cache.get("df")
    print(df)
