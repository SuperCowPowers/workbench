"""Log Utilities"""

import os
import logging
from contextlib import contextmanager


@contextmanager
def log_level(level=logging.WARNING):
    """Temporarily set the workbench log level, restoring it afterwards.

    Unlike silence_logs(), warnings and errors still come through. Useful for
    quieting routine INFO chatter during an interactive operation.

    Args:
        level: The temporary log level (default: logging.WARNING)
    """
    # Skip this if the WORKBENCH_DEBUG environment variable is set to True
    if os.getenv("WORKBENCH_DEBUG", "False").lower() == "true":
        yield
        return

    logger = logging.getLogger("workbench")
    original_level = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(original_level)


@contextmanager
def silence_logs():
    """Be careful, this can be fairly dangerous, as it suppresses errors that are important to see"""

    # Skip this if the WORKBENCH_DEBUG environment variable is set to True
    if os.getenv("WORKBENCH_DEBUG", "False").lower() == "true":
        yield
        return

    # Suppress all logs greater than ERROR
    logger = logging.getLogger("workbench")
    original_level = logger.level
    try:
        logger.setLevel(logging.ERROR + 1)
        yield
    finally:
        logger.setLevel(original_level)


if __name__ == "__main__":
    # Test the log utils functions
    from workbench.utils.workbench_logging import logging_setup

    logging_setup()

    log = logging.getLogger("workbench")
    log.setLevel(logging.DEBUG)
    log.info("You should see me")

    with silence_logs():
        log.info("You should NOT see me")
        log.warning("You should NOT see me")

    log.info("You should see me")
    log.warning("You should see this warning")

    try:
        with silence_logs():
            raise ValueError("This is a test error")
    except ValueError:
        pass
    log.info("You should see me")
