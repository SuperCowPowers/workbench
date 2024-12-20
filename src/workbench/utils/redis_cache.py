import json
import numpy as np
import pandas as pd
import redis
import logging
from datetime import datetime, date

# Local Imports
from workbench.utils.config_manager import ConfigManager
from workbench.utils.json_utils import CustomEncoder, custom_decoder

log = logging.getLogger("workbench")


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

    def __init__(self, expire=None, prefix="", postfix=""):
        """RedisCache Initialization
        Args:
            expire: the number of seconds to keep items in the redis_cache
            prefix: the prefix to use for all keys
            postfix: the postfix to use for all keys
        """
        # Setup instance variables
        self.expire = expire
        self.base_prefix = "workbench:"  # Prefix all keys with the Workbench namespace
        self.prefix = prefix if not prefix or prefix.endswith(":") else prefix + ":"
        self.prefix = self.base_prefix + self.prefix
        self.postfix = postfix if not postfix or postfix.startswith(":") else ":" + postfix

        # Load Redis configuration from the Workbench ConfigManager
        cm = ConfigManager()
        self.host = cm.get_config("REDIS_HOST", "localhost")
        self.port = cm.get_config("REDIS_PORT", 6379)
        self.password = cm.get_config("REDIS_PASSWORD")

        # Attempt to establish a connection to Redis
        log.info(f"Opening Redis connection to: {self.host}:{self.port}...")
        self.redis_db = None
        try:
            # Create a temporary connection to test the connection
            _redis_db = redis.Redis(self.host, port=self.port, password=self.password, socket_timeout=1)
            _redis_db.ping()

            # Now create the actual connection
            self.redis_db = redis.Redis(
                self.host,
                port=self.port,
                password=self.password,
                charset="utf-8",
                decode_responses=True,
                db=0,
            )
            log.info(f"Redis connection success: {self.host}:{self.port}...")
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            log.error(f"Redis Database connection failed: {self.host}:{self.port} - {str(e)}")
            log.error("1. AWS Glue/Lambda: Check VPC settings (Inbound Rules, Security Groups).")
            log.error("2. Local/Notebooks: Check if VPN is active or required for Redis access.")
            log.error("3. Redis Configuration: Ensure Redis server is running and accessible.")

    def check(self):
        return self.redis_db is not None

    def set(self, key, value):
        """Add an item to the redis_cache, all items are JSON serialized
        Args:
               key: item key
               value: the value associated with this key
        """
        self._set(key, json.dumps(value, cls=CustomEncoder))

    def get(self, key):
        """Get an item from the redis_cache, all items are JSON deserialized
        Args:
            key: item key
        Returns:
            the value of the item or None if the item isn't in the redis_cache
        """
        raw_value = self._get(key)
        return json.loads(raw_value, object_hook=custom_decoder) if raw_value else None

    def _set(self, key, value):
        """Internal Method: Add an item to the redis_cache"""
        # Never except a key or value with a 'falsey' value
        if not key or not value:
            return
        self.redis_db.set(self.prefix + str(key) + self.postfix, value, ex=self.expire)

    def _get(self, key):
        """Internal Method: Get an item from the redis_db_cache"""
        if not key:
            return None
        return self.redis_db.get(self.prefix + str(key) + self.postfix)

    def delete(self, key):
        """Delete an item from the redis_cache"""
        # Delete the key and its full key (if either exists)
        self.redis_db.delete(key)
        self.redis_db.delete(self.prefix + str(key) + self.postfix)

    def list_keys(self):
        """List all keys in the redis_cache"""
        return self.redis_db.keys(self.prefix + "*")

    def list_subkeys(self, key):
        """List all sub-keys in the redis_cache"""
        return self.redis_db.keys(self.prefix + str(key) + "*")

    def clear(self, all_keys=False):
        """Clear the redis_cache

        Args:
            all_keys: if True, clear all keys in the redis_cache, otherwise only clear keys with the prefix
        """
        if all_keys:
            print("Clearing ALL Redis Keys...")
            self.redis_db.flushdb()
        else:
            print(f"Clearing {self.prefix} Redis Keys...")
            for key in self.redis_db.scan_iter(self.prefix + "*"):
                self.redis_db.delete(key)

    def dump(self, all_keys=False):
        """Dump the redis_cache (for debugging)

        Args:
           all_keys: if True, print out ALL keys in the redis_cache, otherwise only keys with the prefix
        """
        if all_keys:
            for key in self.redis_db.scan_iter():
                print(key, ":", self.redis_db.get(key))
        else:
            for key in self.redis_db.scan_iter(self.prefix + "*"):
                print(key, ":", self.redis_db.get(key))

    def size(self):
        return self.redis_db.dbsize()

    def get_key_info(self):
        """Get information about all keys in the Redis database

        Returns:
            A list of dictionaries containing key information: name, size, and last modified date.
        """
        keys_info = []
        for key in self.redis_db.scan_iter("*"):
            key_info = {
                "key": key,
                "size": self.redis_db.memory_usage(key),
                "last_modified": self.redis_db.object("idletime", key),
            }
            keys_info.append(key_info)
        return keys_info

    def get_memory_config(self):
        """Get Redis memory usage and configuration settings as a dictionary"""
        info = {}
        try:
            memory_info = self.redis_db.info("memory")
            info["used_memory"] = memory_info.get("used_memory", "N/A")
            info["used_memory_human"] = memory_info.get("used_memory_human", "N/A")
            info["mem_fragmentation_ratio"] = memory_info.get("mem_fragmentation_ratio", "N/A")
            info["maxmemory_policy"] = memory_info.get("maxmemory_policy", "N/A")
        except redis.exceptions.RedisError as e:
            log.error(f"Error retrieving memory info from Redis: {e}")

        try:
            max_memory = self.redis_db.config_get("maxmemory")
            info["maxmemory"] = max_memory.get("maxmemory", "N/A")
        except redis.exceptions.RedisError as e:
            log.error(f"Error retrieving config info from Redis (likely unsupported command): {e}")
            info["maxmemory"] = "Not Available - Command Restricted"

        return info

    def report_memory_config(self):
        """Print Redis memory usage and configuration settings"""
        info = self.get_memory_config()
        print("Redis Memory Usage Report:")
        print(f"\tUsed Memory: {info.get('used_memory_human', 'N/A')} ({info.get('used_memory', 'N/A')} bytes)")
        print(f"\tMemory Fragmentation Ratio: {info.get('mem_fragmentation_ratio', 'N/A')}")
        print(f"\tMax Memory Config: {info.get('maxmemory')}")
        print(f"\tMax Memory Policy: {info.get('maxmemory_policy', 'N/A')}")

    def get_largest_keys(self, n=5):
        """Get the N largest keys in the Redis database by size"""
        keys_info = self.get_key_info()
        largest_keys = sorted(keys_info, key=lambda x: x["size"], reverse=True)[:n]
        return largest_keys

    def report_largest_keys(self, n=5):
        """Print the N largest keys in the Redis database by size"""
        largest_keys = self.get_largest_keys(n)
        print(f"Top {n} largest keys in Redis:")
        for key_info in largest_keys:
            size_mb = key_info["size"] / 1024 / 1024
            days = key_info["last_modified"] // 86400
            print(f"\t{size_mb:.2f}MB {key_info['key']}   ({days} days)")

    def delete_keys_older_than(self, days, dry_run=True):
        """Delete keys in the Redis database that are older than a specified number of days"""
        age_threshold = days * 86400
        for key in self.redis_db.scan_iter("*"):
            last_modified = self.redis_db.object("idletime", key)
            if last_modified and last_modified > age_threshold:
                days_mod = last_modified // 86400
                if dry_run:
                    log.info(f"DRY RUN: {key} ({days_mod} days old)")
                else:
                    self.redis_db.delete(key)
                    log.info(f"Deleted key: {key} ({days_mod} days old)")

        log.info(f"Completed deletion of keys older than {days} days.")

    def __repr__(self):
        if self.redis_db is None:
            return f"RedisCache(Prefix={self.prefix}  Host={self.host}:{self.port}) Failed to Connect!"
        return f"RedisCache(Prefix={self.prefix}  Host={self.host}:{self.port})"


if __name__ == "__main__":
    from pprint import pprint
    import time

    # Create a RedisCache
    my_redis_cache = RedisCache(prefix="test")
    if not my_redis_cache.check():
        print("Redis not available, exiting...")
        exit(1)

    # Test the REPR
    print(my_redis_cache)

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
    assert my_redis_cache.size() >= 5

    # Dump the redis_cache
    my_redis_cache.dump()

    # Test deleting
    my_redis_cache.delete("1")
    assert my_redis_cache.get("1") is None

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

    # Test listing keys
    print("Listing Keys...")
    print(my_redis_cache.list_keys())
    print(my_redis_cache.list_subkeys("foo"))

    # Clear out the Redis Cache (just the test keys)
    my_redis_cache.clear()

    # Test storing numpy data
    data = {"int": np.int64(6), "float": np.float64(6.5), "array": np.array([1, 2, 3])}
    my_redis_cache.set("data", data)
    ret_data = my_redis_cache.get("data")
    pprint(data)
    pprint(ret_data)

    # Test storing datetime data
    data = {"now": datetime.now(), "today": date.today()}
    my_redis_cache.set("data", data)
    ret_data = my_redis_cache.get("data")
    pprint(data)
    pprint(ret_data)

    # Test storing pandas data
    data = {"my_dataframe": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})}
    my_redis_cache.set("data", data)
    ret_data = my_redis_cache.get("data")
    pprint(data)
    pprint(ret_data)

    # Report memory usage and configuration settings
    my_redis_cache.report_memory_config()

    # Report the largest keys in the Redis database
    my_redis_cache.report_largest_keys(5)

    # Delete keys older than 1 day
    my_redis_cache.delete_keys_older_than(1, dry_run=True)
