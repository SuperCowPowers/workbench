"""Setup logging defaults"""
import logging
import os


def logging_setup(level=None):
    logger = logging.getLogger("sageworks")

    # If the logger already has handlers, it's already configured.
    # Just update the level and return.
    if logger.hasHandlers():
        logger.setLevel(level or os.environ.get("LOGLEVEL", "INFO"))
        return

    # If we're here, the logger has not been configured yet.
    logger.setLevel(level or os.environ.get("LOGLEVEL", "INFO"))

    handler = logging.StreamHandler(os.sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == "__main__":
    logging_setup()
    log = logging.getLogger("sageworks")
    log.info("Info: You should see me")
    logging.getLogger("sageworks").setLevel(logging.WARNING)
    log.info("Info: You should NOT see me")
    log.warning("Warning: You should see me")
