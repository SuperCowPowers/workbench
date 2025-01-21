"""Log Utilities"""

import os
import logging
from contextlib import contextmanager
from workbench.utils.workbench_logging import ColoredFormatter


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


def log_theme(theme: str):
    """Update the logging theme dynamically."""
    # Validate the theme
    if theme.lower() not in ["light", "dark"]:
        raise ValueError("Theme must be 'light' or 'dark'")

    # Set the global theme in ColoredFormatter
    ColoredFormatter.set_theme(theme)
    log = logging.getLogger("workbench")

    # Replace the formatter for all handlers
    for handler in log.handlers:
        formatter = handler.formatter
        if formatter and formatter.__class__.__name__ == "ColoredFormatter":
            # Create a new formatter with the updated theme
            new_formatter = ColoredFormatter(
                "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(new_formatter)


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

    # Test the log theme
    log_theme("light")
    print("\n\n\n")
    log.info("Switched to light theme")
    log.debug("This should be a muted color")
    log.trace("Trace color should stand out from debug")
    log.info("This should be a nice color")
    log.important("Important color should stand out from info")
    log.warning("This should be a color that attracts attention")
    log.monitor("This is a monitor message")
    log.error("This should be a bright color")
    log.critical("This should be an alert color")

    log_theme("dark")
    print("\n\n\n")
    log.info("Switched to light theme")
    log.debug("This should be a muted color")
    log.trace("Trace color should stand out from debug")
    log.info("This should be a nice color")
    log.important("Important color should stand out from info")
    log.warning("This should be a color that attracts attention")
    log.monitor("This is a monitor message")
    log.error("This should be a bright color")
    log.critical("This should be an alert color")
