# Description: Give a report on the redis server data size and last modified time.
import argparse
import logging
from sageworks.utils.redis_cache import RedisCache

# Set up logging
log = logging.getLogger("sageworks")


def redis_report(n: int = 20):
    """Print the top N Redis Data sizes and last modified time."""
    redis_cache = RedisCache()

    # Check if Redis is available
    if not redis_cache.check():
        log.error("Redis not available, exiting...")
        return

    # Report memory usage and configuration settings
    redis_cache.report_memory_config()

    # Report the largest keys in the Redis database
    redis_cache.report_largest_keys(n)


def parse_args():
    parser = argparse.ArgumentParser(description="Redis Report.")
    parser.add_argument("--n", type=int, default=20, help="Top N Redis Data sizes Default is 20.")
    return parser.parse_args()


def main():
    args = parse_args()
    redis_report(args.n)


if __name__ == "__main__":
    main()
