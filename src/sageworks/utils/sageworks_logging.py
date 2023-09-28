"""Setup logging defaults"""
import logging
import os
from functools import lru_cache


@lru_cache(maxsize=None)  # Cache the logging setup
def logging_setup():
    logger = logging.getLogger("sageworks")
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    handler = logging.StreamHandler(os.sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
