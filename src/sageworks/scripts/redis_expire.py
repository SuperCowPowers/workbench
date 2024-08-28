import argparse
import logging
from sageworks.utils.redis_cache import RedisCache

# Set up logging
log = logging.getLogger("sageworks")


def main(days, dry_run):
    # Create an instance of RedisCache
    redis_cache = RedisCache()

    # Check if Redis is available
    if not redis_cache.check():
        log.error("Redis not available, exiting...")
        return

    # Call the delete_keys_older_than method with the specified parameters
    redis_cache.delete_keys_older_than(days=days, dry_run=dry_run)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Delete keys older than N days from Redis.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days after which keys are deleted (default: 30)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually delete keys (the default is to show what would be deleted)",
    )

    args = parser.parse_args()

    # Execute main function
    main(days=args.days, dry_run=args.dry_run)
