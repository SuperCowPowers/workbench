import os
import logging


def logging_setup():
    log = logging.getLogger("sageworks")
    if not log.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
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

        log.info("SageWorks Logging Setup Complete...")


if __name__ == "__main__":
    # Uncomment to test the SAGEWORKS_DEBUG env variable
    # os.environ["SAGEWORKS_DEBUG"] = "True"

    logging_setup()
    my_log = logging.getLogger("sageworks")
    my_log.info("Info: You should see me")
    my_log.debug("Debug: You should see me only if SAGEWORKS_DEBUG is True")
    logging.getLogger("sageworks").setLevel(logging.WARNING)
    my_log.info("Info: You should NOT see me")
    my_log.warning("Warning: You should see me")
