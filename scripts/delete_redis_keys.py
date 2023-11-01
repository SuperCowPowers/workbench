import redis


def delete_keys_by_pattern(redis_host, redis_port, pattern):
    # Connect to the Redis instance
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    # Use the SCAN command to find all keys matching the pattern
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=1000)
        for key in keys:
            print(f"Deleting {key}...")
            r.unlink(key)
        if cursor == 0:
            break


if __name__ == "__main__":
    redis_host = "sageworksredis.qo8vb5.0001.use1.cache.amazonaws.com"
    redis_port = 6379
    pattern = "sageworks:RowStorage:*"

    delete_keys_by_pattern(redis_host, redis_port, pattern)
