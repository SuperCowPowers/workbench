"""A Redis Database Cache Class"""
import os
import json

import redis
import logging
from datetime import datetime, date

# Local Imports
from sageworks.utils.iso_8601 import datetime_to_iso8601, iso8601_to_datetime
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class RedisCache:
    """A Redis Database Cache Class
    Usage:
         redis_cache = RedisCache(expire=10)
         redis_cache.set('foo', 'bar')
         redis_cache.get('foo')
         > bar
         time.sleep(11)
         redis_cache.get('foo')
         > None
         redis_cache.clear()
    """

    # Setup logger (class attribute)
    log = logging.getLogger("sageworks")

    # Try to read Redis configuration from environment variables
    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD", None)

    # Open the Redis connection (class object)
    log.info(f"Opening Redis connection to: {host}:{port}...")
    redis_db = None
    try:
        # Create a temporary connection to test the connection
        _redis_db = redis.Redis(host, port=port, password=password, socket_timeout=1)
        _redis_db.ping()

        # Now create the actual connection
        redis_db = redis.Redis(
            host,
            port=port,
            password=password,
            charset="utf-8",
            decode_responses=True,
            db=0,
        )
        log.info(f"Redis connection success: {host}:{port}...")
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        msg = f"Could not connect to Redis Database: {host}:{port}..."
        log.warning(msg)

    @classmethod
    def check(cls):
        return cls.redis_db is not None

    def __init__(self, expire=None, prefix="", postfix=""):  # No expiration, standard 0 db, no postfix on keys
        """RedisCache Initialization"""

        # Setup instance variables
        self.expire = expire
        self.base_prefix = "sageworks:"  # Prefix all keys with the SageWorks namespace
        self.prefix = prefix if not prefix or prefix.endswith(":") else prefix + ":"
        self.prefix = self.base_prefix + self.prefix
        self.postfix = postfix if not postfix or postfix.startswith(":") else ":" + postfix

    def set(self, key, value):
        """Add an item to the redis_cache, all items are JSON serialized
        Args:
               key: item key
               value: the value associated with this key
        """
        self._set(key, json.dumps(value, default=self.serialize_datetime))

    def get(self, key):
        """Get an item from the redis_cache, all items are JSON deserialized
        Args:
            key: item key
        Returns:
            the value of the item or None if the item isn't in the redis_cache
        """
        raw_value = self._get(key)
        return json.loads(raw_value, object_pairs_hook=self.deserialize_datetime) if raw_value else None

    def _set(self, key, value):
        """Internal Method: Add an item to the redis_cache"""
        # Never except a key or value with a 'falsey' value
        if not key or not value:
            return

        # Set the value in redis with optional postfix and the expire
        self.redis_db.set(self.prefix + str(key) + self.postfix, value, ex=self.expire)

    def _get(self, key):
        """Internal Method: Get an item from the redis_db_cache"""
        if not key:
            return None

        # Get the value in redis with optional postfix
        return self.redis_db.get(self.prefix + str(key) + self.postfix)

    @classmethod
    def clear(cls):
        """Clear the redis_cache"""
        print("Clearing Redis Cache...")
        cls.redis_db.flushall()

    @classmethod
    def dump(cls):
        """Dump the redis_cache (for debugging)"""
        for key in cls.redis_db.scan_iter():
            print(key, ":", cls.redis_db.get(key))

    @classmethod
    def size(cls):
        return cls.redis_db.dbsize()

    @staticmethod
    def serialize_datetime(obj):
        """JSON serializer for datetime objects"""
        if isinstance(obj, (datetime, date)):
            return datetime_to_iso8601(obj)
        raise TypeError("Type %s not serializable" % type(obj))

    @staticmethod
    def deserialize_datetime(pairs):
        """JSON deserializer for datetime objects"""
        d = {}
        for k, v in pairs:
            if isinstance(v, str):
                try:
                    d[k] = iso8601_to_datetime(v)
                except ValueError:
                    d[k] = v
            else:
                d[k] = v
        return d


if __name__ == "__main__":
    """Exercise the RedisCache class"""
    import time

    # Create a RedisCache
    my_redis_cache = RedisCache(prefix="test")
    if not my_redis_cache.check():
        print("Redis not available, exiting...")
        exit(1)

    # Delete anything in the test database
    my_redis_cache.clear()

    # Test storage
    my_redis_cache.set("foo", "bar")
    assert my_redis_cache.get("foo") == "bar"
    my_redis_cache.dump()
    my_redis_cache.clear()

    # Create the RedisCache with an expiry
    my_redis_cache = RedisCache(expire=1, prefix="test")

    # Test expire
    my_redis_cache.set("foo", "bar")
    time.sleep(1.1)
    assert my_redis_cache.get("foo") is None

    # Make sure size is working
    for i in range(1, 6):
        my_redis_cache.set(str(i), i)
    print(my_redis_cache.size())
    assert my_redis_cache.size() == 5

    # Dump the redis_cache
    my_redis_cache.dump()

    # Test storing 'falsey' values
    my_redis_cache.set(0, "foo")
    my_redis_cache.set(0, "bar")
    my_redis_cache.set(None, "foo")
    my_redis_cache.set("", None)
    assert my_redis_cache.get("") is None
    assert my_redis_cache.get(None) is None
    assert my_redis_cache.get(0) is None
    my_redis_cache.set("blah", None)
    assert my_redis_cache.get("blah") is None

    # Test storing complex data
    data = {"foo": {"bar": 5, "baz": "blah"}}
    my_redis_cache.set("data", data)
    ret_data = my_redis_cache.get("data")
    assert data == ret_data

    # Test keys with postfix
    my_redis_cache = RedisCache(prefix="test", postfix="fresh")
    my_redis_cache.set("foo", "bar")
    assert my_redis_cache.get("foo") == "bar"

    # Clear out the Redis Cache
    my_redis_cache.clear()
    assert my_redis_cache.size() == 0
