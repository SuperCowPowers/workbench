import argparse
import redis
import datetime


def redis_reaper(host, port, size_limit=1048576, expiration_days=30):
    """Reap keys from Redis based on size and expiration date.

    Args:
        host (str): Redis host.
        port (int): Redis port.
        size_limit (int): Size limit in bytes (default: 1MB).
        expiration_days (int): Number of days before a key is considered expired (default: 30 days).
    """
    # Connect to Redis
    client = redis.Redis(host=host, port=port)

    # Current time for expiration check
    now = datetime.datetime.utcnow()
    keys_deleted = 0

    # Iterate through all keys
    for key in client.scan_iter():
        # Check the size of the value
        size = client.memory_usage(key) or 0
        if size > size_limit:
            print(f"Deleting key {key} (size: {size} bytes)")
            # client.delete(key)
            keys_deleted += 1
            continue

        # Check the last modified/created time
        ttl = client.ttl(key)
        if ttl == -1:  # Skip keys that don't have an expiration set
            continue
        creation_time = now - datetime.timedelta(seconds=ttl)
        if creation_time < now - datetime.timedelta(days=expiration_days):
            print(f"Deleting key {key} (expired: {creation_time})")
            # client.delete(key)
            keys_deleted += 1

    print(f"Total keys deleted: {keys_deleted}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Redis Reaper: Delete keys based on size and age.")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host")
    parser.add_argument("-p", "--port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--size-limit", type=int, default=1048576, help="Size limit in bytes (default: 1MB)")
    parser.add_argument("--expire", type=int, default=7, help="Expiration days (default: 7 days)")
    args = parser.parse_args()

    # Run the Redis Reaper
    redis_reaper(args.host, args.port, args.size_limit, args.expire)
