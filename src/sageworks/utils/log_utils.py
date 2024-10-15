"""Log Utilities"""

import logging
from contextlib import contextmanager


@contextmanager
def quiet_execution():
    logger = logging.getLogger("sageworks")
    original_level = logger.level
    try:
        logger.setLevel(logging.CRITICAL + 1)
        yield
    except Exception as e:
        logger.setLevel(logging.WARNING)  # Temporarily lower log level
        logger.warning(f"Exception occurred during deletion: {e}")
        logger.warning("In general this warning can be ignored :)")
    finally:
        logger.setLevel(original_level)


if __name__ == "__main__":
    # Test the log utils functions

    log = logging.getLogger("sageworks")
    log.setLevel(logging.INFO)
    log.info("You should see me")

    with quiet_execution():
        log.info("You should NOT see me")
        log.warning("You should NOT see me")

    log.info("You should see me")
    log.warning("You should see this warning")

    with quiet_execution():
        raise ValueError("This is a test error")
    log.info("You should see me")
