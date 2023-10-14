"""Setup SageWorks logging defaults"""
import logging


def logging_setup():
    log = logging.getLogger('sageworks')
    if not log.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.info("SageWorks Logging Setup Complete...")


if __name__ == "__main__":
    logging_setup()
    my_log = logging.getLogger("sageworks")
    my_log.info("Info: You should see me")
    logging.getLogger("sageworks").setLevel(logging.WARNING)
    my_log.info("Info: You should NOT see me")
    my_log.warning("Warning: You should see me")