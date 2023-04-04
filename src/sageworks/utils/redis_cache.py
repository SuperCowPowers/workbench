"""A Redis Database Cache Class"""
import json
import redis
import time
import logging


class RedisCache(object):
    """A Redis Database Cache Class
       Usage:
            redis_cache = RedisCache(expire=10)
            redis_cache.set('foo', 'bar')
            redis_cache.get('foo')
            >>> bar
            time.sleep(11)
            redis_cache.get('foo')
            >>> None
            redis_cache.clear()
    """
    # Setup logger (class attribute)
    log = logging.getLogger(__name__)

    def __init__(self, expire=None, db=0):  # No expiration and standard 0 db
        """RedisCache Initialization"""

        # Quick connection test
        self.ping()

        # Setup redis handles
        self.expire = expire
        self.host, self.port, self.password = self.get_redis_config()
        self.redis = redis.Redis(self.host, port=self.port, password=self.password,
                                 charset='utf-8', decode_responses=True, db=db)

    @classmethod
    def ping(cls):
        """Helper method to quickly ping the Redis database"""
        host, port, password = cls.get_redis_config()
        r = redis.Redis(host, port=port, password=password, socket_timeout=1)

        # Test the connection to the Redis database
        cls.log.info('Opening Redis connection to: {:s}...'.format(host))
        try:
            r.ping()
            cls.log.info('Redis connection success: {:s}...'.format(host))
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            msg = 'Could not connect to Redis Database -- host:{:s}  port:{:d}'.format(host, port)
            cls.log.critical(msg)
            raise RuntimeError(msg)

    @staticmethod
    def get_redis_config():
        """Grab the Redis config values for host, port, pass"""
        host = 'localhost'
        port = 6379
        password = None
        return host, port, password

    def set(self, key, value):
        """Add an item to the redis_cache, all items are JSON serialized
        Args:
               key: item key
               value: the value associated with this key
        """
        self._set(key, json.dumps(value))

    def get(self, key):
        """Get an item from the redis_cache, all items are JSON deserialized
           Args:
               key: item key
           Returns:
               the value of the item or None if the item isn't in the redis_cache
        """
        raw_value = self._get(key)
        return json.loads(raw_value) if raw_value else None

    def _set(self, key, value):
        """Internal Method: Add an item to the redis_cache"""
        # Never except a key or value with a 'falsey' value
        if not key or not value:
            return

        # Set the value in redis with the expire
        self.redis.set(key, value, ex=self.expire)

    def _get(self, key):
        """Internal Method: Get an item from the redis_cache"""
        if not key:
            return None
        return self.redis.get(key)

    def clear(self):
        """Clear the redis_cache"""
        print('Clearing Redis Cache...')
        self.redis.flushall()

    def dump(self):
        """Dump the redis_cache (for debugging)"""
        for key in self.redis.scan_iter():
            print(key, ':', self.get(key))

    @property
    def size(self):
        return self.redis.dbsize()


if __name__ == '__main__':
    """Exercise the RedisCache class"""

    # Use a test database
    test_db = 15

    # Test Redis ConnectionError
    RedisCache.ping()

    # Create the RedisCache
    my_redis_cache = RedisCache()

    # Delete anything in the test database
    my_redis_cache.clear()

    # Test storage
    my_redis_cache.set('foo', 'bar')
    assert my_redis_cache.get('foo') == 'bar'
    my_redis_cache.clear()

    # Create the RedisCache with an expire
    my_redis_cache = RedisCache(expire=1)

    # Test expire
    my_redis_cache.set('foo', 'bar')
    time.sleep(1.1)
    assert my_redis_cache.get('foo') is None

    # Make sure size is working
    for i in range(1, 6):
        my_redis_cache.set(str(i), i)
    print(my_redis_cache.size)
    assert my_redis_cache.size == 5

    # Dump the redis_cache
    my_redis_cache.dump()

    # Test storing 'falsey' values
    my_redis_cache.set(0, 'foo')
    my_redis_cache.set(0, 'bar')
    my_redis_cache.set(None, 'foo')
    my_redis_cache.set('', None)
    assert my_redis_cache.get('') is None
    assert my_redis_cache.get(None) is None
    assert my_redis_cache.get(0) is None
    my_redis_cache.set('blah', None)
    assert my_redis_cache.get('blah') is None

    # Test storing complex data
    data = {'foo': {'bar': 5, 'baz': 'blah'}}
    my_redis_cache.set('data', data)
    ret_data = my_redis_cache.get('data')
    assert data == ret_data

    my_redis_cache.clear()
    assert my_redis_cache.size == 0
