"""Cache class for key/value pairs"""

import time
from collections import OrderedDict
import atexit


class Cache:
    """In process memory cache. Not thread safe.
    Usage:
         cache = Cache(max_size=5, expire=10)
         cache.set('foo', 'bar')
         cache.get('foo')
         > bar
         time.sleep(11)
         cache.get('foo')
         > None
         cache.clear()
    """

    def __init__(self, max_size=10000, expire=None, prefix="", postfix=""):
        """Cache Initialization"""
        self.store = OrderedDict()
        self.max_size = max_size
        self.expire = expire
        self._compression_timer = 600
        self._last_compression = time.time()
        self.prefix = prefix if not prefix or prefix.endswith(":") else prefix + ":"
        self.postfix = postfix if not postfix or postfix.startswith(":") else ":" + postfix

        # Try to do cleanup/serialization at exit
        atexit.register(self.cleanup)

    def _get_full_key(self, key):
        return f"{self.prefix}{key}{self.postfix}"

    @staticmethod
    def check():
        """Check the status of this cache"""
        return True  # I'm doing great, thanks for asking

    def set(self, key, value):
        """Add an item to the cache
        Args:
               key: item key
               value: the value associated with this key
        """
        self._check_limit()
        actual_key = self._get_full_key(key)
        _expire = time.time() + self.expire if self.expire else None
        self.store[actual_key] = (value, _expire)

    def get(self, key):
        """Get an item from the cache
        Args:
            key: item key
        Returns:
            the value of the item or None if the item isn't in the cache
        """
        actual_key = self._get_full_key(key)
        data = self.store.get(actual_key)
        if not data:
            return None
        value, expire = data
        if expire and time.time() > expire:
            del self.store[actual_key]
            return None
        return value

    def delete(self, key):
        # Delete the key and its full key (if either exists)
        if key in self.store:
            del self.store[key]
        full_key = self._get_full_key(key)
        if full_key in self.store:
            del self.store[full_key]

    def list_keys(self):
        """List all keys in the cache"""
        return list(self.store.keys())

    def list_subkeys(self, key):
        """List all sub-keys in the cache"""
        return [k for k in self.store.keys() if k.startswith(f"{self.prefix}{key}")]

    def clear(self):
        """Clear the cache"""
        self.store = OrderedDict()

    def dump(self):
        """Dump the cache (for debugging)"""
        for key in self.store.keys():
            print(key, ":", self.store.get(key))

    @property
    def size(self):
        return len(self.store)

    def cleanup(self):
        """Cleanup the cache (if we need to)"""
        pass

    def _check_limit(self):
        """Internal method: check if current cache size exceeds maximum cache
        size and pop the oldest item in this case"""

        # First compress
        self._compress()

        # Then check the max size
        if len(self.store) >= self.max_size:
            self.store.popitem(last=False)  # FIFO

    def _compress(self):
        """Internal method to compress the cache. This method will
        expire any old items in the cache, making the cache smaller"""

        # Don't compress too often
        now = time.time()
        if self._last_compression + self._compression_timer < now:
            self._last_compression = now
            for key in list(self.store.keys()):
                self.get(key)

    def __repr__(self):
        return f"MemoryCache(Prefix={self.prefix})"


if __name__ == "__main__":
    """Exercise the Cache class"""

    # Create the Cache
    my_cache = Cache(max_size=5, expire=1)
    my_cache.set("foo", "bar")

    # Test storage
    assert my_cache.get("foo") == "bar"

    # Test expire
    time.sleep(1.1)
    assert my_cache.get("foo") is None

    # Test max_size
    my_cache = Cache(max_size=5)
    for i in range(6):
        my_cache.set(str(i), i)

    # So the '0' key should no longer be there FIFO
    assert my_cache.get("0") is None
    assert my_cache.get("5") is not None

    # Test listing keys
    print("Listing Keys...")
    print(my_cache.list_keys())
    print(my_cache.list_subkeys("foo"))

    # Make sure size is working
    assert my_cache.size == 5

    # Dump the cache
    my_cache.dump()

    # Test deleting
    my_cache.delete("1")
    assert my_cache.get("1") is None

    # Test storing 'null' values
    my_cache.set(0, "foo")
    my_cache.set(0, "bar")
    my_cache.set(None, "foo")
    my_cache.set("", None)
    assert my_cache.get("") is None
    assert my_cache.get(None) == "foo"
    assert my_cache.get(0) == "bar"

    # Test the cache compression
    my_cache = Cache(max_size=5, expire=1)
    for i in range(5):
        my_cache.set(str(i), i)
    my_cache._compression_timer = 1
    assert my_cache.size == 5

    # Make sure compression is working
    time.sleep(1.1)
    my_cache._compress()
    assert my_cache.size == 0

    # Also make sure compression call is throttled
    my_cache._compress()  # Should not output a compression message

    # Test prefix/postfix
    my_cache = Cache(max_size=5, prefix="pre", postfix="post")
    my_cache.set("foo", "bar")
    assert my_cache.get("foo") == "bar"
    my_cache.dump()
