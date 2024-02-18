import os
import sys
import logging
from collections import defaultdict
import time


# Check if we're running on Docker
# AWS Cloud Watch doesn't like ANSI escape codes or custom log levels
from sageworks.utils.docker_utils import running_on_ecs

on_ecs = running_on_ecs()


class ThrottlingFilter(logging.Filter):
    def __init__(self, rate_seconds=60):
        super().__init__()
        self.rate_seconds = rate_seconds
        self.last_log_times = defaultdict(lambda: 0)

    def filter(self, record):
        # Get the message and last log time for this message
        message = str(record.msg)
        last_log_time = self.last_log_times[message]

        # Return True if this message should be logged (i.e. it's been long enough since the last time)
        current_time = time.time()
        if current_time - last_log_time > self.rate_seconds:
            self.last_log_times[message] = current_time
            return True

        # Filter out this message
        return False


# Define TRACE level
# Note: see https://docs.python.org/3/library/logging.html#logging-levels
TRACE_LEVEL_NUM = 15  # Between DEBUG and INFO
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


# Define IMPORTANT level
# Note: see https://docs.python.org/3/library/logging.html#logging-levels
IMPORTANT_LEVEL_NUM = 25  # Between INFO and WARNING
logging.addLevelName(IMPORTANT_LEVEL_NUM, "IMPORTANT")


def important(self, message, *args, **kws):
    if self.isEnabledFor(IMPORTANT_LEVEL_NUM):
        self._log(IMPORTANT_LEVEL_NUM, message, args, **kws)


# Add the trace and important level to the logger
logging.Logger.trace = trace
logging.Logger.important = important


# Define a ColoredFormatter
class ColoredFormatter(logging.Formatter):
    COLORS_DARK_THEME = {
        "DEBUG": "\x1b[38;5;245m",  # LightGrey
        "TRACE": "\x1b[38;5;141m",  # LightPurple
        "INFO": "\x1b[38;5;69m",  # LightBlue
        "IMPORTANT": "\x1b[38;5;113m",  # LightGreen
        "WARNING": "\x1b[38;5;220m",  # DarkYellow
        "ERROR": "\x1b[38;5;208m",  # Orange
        "CRITICAL": "\x1b[38;5;198m",  # Red
    }
    COLORS_LIGHT_THEME = {
        "DEBUG": "\x1b[38;5;21m",  # Blue
        "TRACE": "\x1b[38;5;91m",  # Purple
        "INFO": "\x1b[38;5;22m",  # Green
        "IMPORTANT": "\x1b[38;5;178m",  # Lime
        "WARNING": "\x1b[38;5;94m",  # DarkYellow
        "ERROR": "\x1b[38;5;166m",  # Orange
        "CRITICAL": "\x1b[38;5;124m",  # Red
    }
    COLORS = COLORS_DARK_THEME

    RESET = "\x1b[0m"

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.RESET)}{log_message}{self.RESET}"


def logging_setup(color_logs=True):
    """Setup the logging for the application.""

    Args:
        color_logs (bool, optional): Whether to colorize the logs. Defaults to True.
    """

    # Cloud Watch doesn't like colors
    if on_ecs:
        color_logs = False
    log = logging.getLogger("sageworks")

    # Turn off propagation to root logger
    log.propagate = False

    # Remove any existing handlers
    while log.handlers:
        log.info("Removing log handler...")
        log.removeHandler(log.handlers[0])

    # Setup new handler
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = (
        ColoredFormatter(
            "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if color_logs
        else logging.Formatter(
            "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # Setup logging level
    debug_env = os.getenv("SAGEWORKS_DEBUG", "False")
    if debug_env.lower() == "true":
        log.setLevel(logging.DEBUG)
        log.debug("Debugging enabled via SAGEWORKS_DEBUG environment variable.")
    else:
        log.setLevel(logging.INFO)
        throttle_filter = ThrottlingFilter(rate_seconds=5)
        handler.addFilter(throttle_filter)

    # Suppress specific logger
    logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

    # Logging setup complete
    log.info("SageWorks Logging Setup Complete...")


if __name__ == "__main__":
    # Uncomment to test the SAGEWORKS_DEBUG env variable
    # os.environ["SAGEWORKS_DEBUG"] = "True"

    logging_setup()
    my_log = logging.getLogger("sageworks")
    my_log.info("You should see me")
    my_log.debug("You should see me only if SAGEWORKS_DEBUG is True")
    logging.getLogger("sageworks").setLevel(logging.WARNING)
    my_log.info("You should NOT see me")
    my_log.warning("You should see this warning")

    # Test out ALL the colors
    logging.getLogger("sageworks").setLevel(logging.DEBUG)
    my_log.debug("This should be a muted color")
    my_log.trace("Trace color should stand out from debug")
    my_log.info("This should be a nice color")
    my_log.important("Important color should stand out from info")
    my_log.warning("This should be a color that attracts attention")
    my_log.error("This should be a bright color")
    my_log.critical("This should be an alert color")
