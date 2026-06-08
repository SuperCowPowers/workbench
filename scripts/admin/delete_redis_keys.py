"""Delete Redis keys matching a pattern from the Workbench cache.

Targets the Redis cluster of the active Workbench config (``REDIS_HOST`` /
``REDIS_PORT`` from ``WORKBENCH_CONFIG``); override with ``--redis-host`` /
``--redis-port``. Defaults to clearing the RowStorage keyspace.

    WORKBENCH_CONFIG=/path/to/config.json python delete_redis_keys.py
    python delete_redis_keys.py --pattern "workbench:SomeOther:*"
"""

import argparse

import redis

from workbench.utils.config_manager import ConfigManager


def delete_keys_by_pattern(redis_host: str, redis_port: int, pattern: str) -> None:
    r = redis.Redis(host=redis_host, port=int(redis_port), decode_responses=True)
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=1000)
        for key in keys:
            print(f"Deleting {key}...")
            r.unlink(key)
        if cursor == 0:
            break


if __name__ == "__main__":
    cm = ConfigManager()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="workbench:RowStorage:*", help="Key pattern to delete")
    parser.add_argument(
        "--redis-host", default=cm.get_config("REDIS_HOST"), help="Redis host (default: config REDIS_HOST)"
    )
    parser.add_argument(
        "--redis-port", default=cm.get_config("REDIS_PORT", "6379"), help="Redis port (default: config REDIS_PORT)"
    )
    args = parser.parse_args()

    if not args.redis_host:
        raise SystemExit("No Redis host: set REDIS_HOST in your Workbench config or pass --redis-host.")

    print(f"Deleting keys matching '{args.pattern}' on {args.redis_host}:{args.redis_port}")
    delete_keys_by_pattern(args.redis_host, args.redis_port, args.pattern)
