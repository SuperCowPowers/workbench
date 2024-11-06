"""Log Utilities"""

import logging
from contextlib import contextmanager


@contextmanager
def silence_logs():
    """Be careful, this can be fairly dangerous, as it suppresses errors that are important to see"""
    logger = logging.getLogger("sageworks")
    original_level = logger.level
    try:
        logger.setLevel(logging.ERROR + 1)
        yield
    finally:
        logger.setLevel(original_level)


if __name__ == "__main__":
    # Test the log utils functions

    log = logging.getLogger("sageworks")
    log.setLevel(logging.INFO)
    log.info("You should see me")

    with silence_logs():
        log.info("You should NOT see me")
        log.warning("You should NOT see me")

    log.info("You should see me")
    log.warning("You should see this warning")

    with silence_logs():
        raise ValueError("This is a test error")
    log.info("You should see me")
