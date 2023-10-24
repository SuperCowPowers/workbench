import os
import sys
import logging


# Define TRACE level
# Note: see https://docs.python.org/3/library/logging.html#logging-levels
TRACE_LEVEL_NUM = 25  # Between INFO and WARNING
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


# Add the trace level to the logger
logging.Logger.trace = trace


# Define a ColoredFormatter
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'TRACE': '\033[38;5;141m',  # LightPurple
        'DEBUG': '\033[38;5;111m',  # LightBlue
        'INFO': '\033[38;5;113m',   # LightGreen
        'WARNING': '\033[38;5;220m',  # Yellow
        'ERROR': '\033[38;5;202m',  # Orange?
        'CRITICAL': '\033[38;5;196m',  # Red
    }

    RESET = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.RESET)}{log_message}{self.RESET}"


def logging_setup(color_logs=True):
    log = logging.getLogger("sageworks")
    if not log.hasHandlers():
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = ColoredFormatter(
            "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ) if color_logs else logging.Formatter(
            "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

        # Check for SAGEWORKS_DEBUG environment variable
        debug_env = os.getenv("SAGEWORKS_DEBUG", "False")
        if debug_env.lower() == "true":
            log.setLevel(logging.DEBUG)
            log.debug("Debugging enabled via SAGEWORKS_DEBUG environment variable.")
        else:
            log.setLevel(logging.INFO)

        # Sagemaker continuously complains about config, so we'll suppress it
        logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

        # All done
        log.info("SageWorks Logging Setup Complete...")


if __name__ == "__main__":
    import time
    # Uncomment to test the SAGEWORKS_DEBUG env variable
    # os.environ["SAGEWORKS_DEBUG"] = "True"

    logging_setup()
    my_log = logging.getLogger("sageworks")
    my_log.info("You should see me")
    my_log.debug("You should see me only if SAGEWORKS_DEBUG is True")
    logging.getLogger("sageworks").setLevel(logging.WARNING)
    my_log.info("You should NOT see me")
    my_log.warning("You should see me")

    # Test out ALL the colors
    logging.getLogger("sageworks").setLevel(logging.DEBUG)
    my_log.debug("\n\nColorized Logging...")
    my_log.debug("This should be a muted color")
    my_log.info("This should be a nice color")
    my_log.trace("This color should stand out from debug and info")
    my_log.warning("This should be a color that attracts attention")
    my_log.error("This should be a bright, alert color")
    my_log.critical("This should be a bright, alert color")
