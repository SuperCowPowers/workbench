import argparse
import redis
import heapq


def redis_reaper(host, port, size_limit=512000, expire=7, batch_size=1000):
    """Reap keys from Redis based on size and age.

    Args:
        host (str): Redis host.
        port (int): Redis port.
        size_limit (int): Size limit in bytes (default: 500KB).
        expire (int): Number of days before a key is considered expired (default: 7 days).
        batch_size (int): Number of keys to process per pipeline batch (default: 1000).
    """
    client = redis.Redis(host=host, port=port)
    total_keys = client.dbsize()
    keys_deleted = 0
    expire_seconds = expire * 86400
    largest_keys = []  # min-heap for top 5
    keys_to_delete = []

    # Process keys in batches using pipelines
    batch = []
    for key in client.scan_iter(count=batch_size):
        batch.append(key)
        if len(batch) >= batch_size:
            keys_deleted += process_batch(client, batch, size_limit, expire_seconds, largest_keys, keys_to_delete)
            batch = []

    # Process remaining keys
    if batch:
        keys_deleted += process_batch(client, batch, size_limit, expire_seconds, largest_keys, keys_to_delete)

    # Batch delete all marked keys
    if keys_to_delete:
        client.delete(*keys_to_delete)

    # Report
    print(f"\nTotal keys: {total_keys}")
    print("\nTop 5 largest keys (by size):")
    for size, key in sorted(largest_keys, reverse=True):
        print(f"Key: {key}, Size: {size} bytes")
    print(f"\nTotal keys deleted: {keys_deleted}")


def process_batch(client, keys, size_limit, expire_seconds, largest_keys, keys_to_delete):
    """Process a batch of keys using pipeline for efficiency."""
    # Pipeline to get memory usage and idle time for all keys
    pipe = client.pipeline(transaction=False)
    for key in keys:
        pipe.memory_usage(key)
        pipe.object("idletime", key)
    results = pipe.execute()

    deleted = 0
    for i, key in enumerate(keys):
        size = results[i * 2] or 0
        idle_time = results[i * 2 + 1]

        # Track largest keys
        heapq.heappush(largest_keys, (size, key))
        if len(largest_keys) > 5:
            heapq.heappop(largest_keys)

        # Check deletion criteria
        if size > size_limit:
            print(f"Deleting key {key} (size: {size} bytes)")
            keys_to_delete.append(key)
            deleted += 1
        elif idle_time is not None and idle_time > expire_seconds:
            print(f"Deleting key {key} (age: {idle_time / 86400:.2f} days)")
            keys_to_delete.append(key)
            deleted += 1
        elif b"Crashed" in key:
            print(f"Deleting key {key} (contains 'Crashed')")
            keys_to_delete.append(key)
            deleted += 1

    return deleted


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Redis Reaper: Delete keys based on size and age.")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host (or host:port)")
    parser.add_argument("-p", "--port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--size-limit", type=int, default=1048576, help="Size limit in bytes (default: 1MB)")
    parser.add_argument("--expire", type=int, default=7, help="Expiration days (default: 7 days)")
    args = parser.parse_args()

    # Parse host:port if provided
    host, port = (args.host.split(":") + [args.port])[:2]
    port = int(port)

    # Run the Redis Reaper
    redis_reaper(host, port, args.size_limit, args.expire)
