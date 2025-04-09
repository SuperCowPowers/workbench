import argparse
import redis
import heapq


def redis_reaper(host, port, size_limit=512000, expire=7):
    """Reap keys from Redis based on size and age.

    Args:
        host (str): Redis host.
        port (int): Redis port.
        size_limit (int): Size limit in bytes (default: 500KB).
        expire (int): Number of days before a key is considered expired (default: 7 days).
    """
    # Connect to Redis
    client = redis.Redis(host=host, port=port)
    total_keys = client.dbsize()
    keys_deleted = 0

    # Use a min-heap to track the 5 largest sizes
    largest_keys = []

    # Iterate through all keys
    for key in client.scan_iter():
        # Check the size of the value
        size = client.memory_usage(key) or 0

        # Track the largest keys
        heapq.heappush(largest_keys, (size, key))
        if len(largest_keys) > 5:
            heapq.heappop(largest_keys)

        if size > size_limit:
            print(f"Deleting key {key} (size: {size} bytes)")
            client.delete(key)
            keys_deleted += 1
            continue

        # Check if the key has a last modified timestamp
        key_info = client.object("idletime", key)
        if key_info is not None:
            key_age_days = key_info / (60 * 60 * 24)
            if key_age_days > expire:
                print(f"Deleting key {key} (age: {key_age_days:.2f} days)")
                client.delete(key)
                keys_deleted += 1

    # Report
    print(f"\nTotal keys: {total_keys}")
    print("\nTop 5 largest keys (by size):")
    for size, key in sorted(largest_keys, reverse=True):
        print(f"Key: {key}, Size: {size} bytes")

    print(f"\nTotal keys deleted: {keys_deleted}")


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
