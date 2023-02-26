"""Setup logging defaults"""
import logging
import os


def logging_setup():
    """This will setup the default formatters and handlers for logging"""
    logging.basicConfig(stream=os.sys.stdout,
                        level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Make boto be more quiet
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
