"""Tests for the AWS Service Broker"""

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_broker import (
    AWSServiceBroker,
    ServiceCategory,
)


def test():
    """Tests for the AWS Service Broker"""
    from pprint import pprint

    # Invoke the AWS Service Broker
    aws_broker = AWSServiceBroker()

    # Get the Metadata for various categories
    for my_category in ServiceCategory:
        print(f"{my_category}:")
        pprint(aws_broker.get_metadata(my_category))

    # Get the Metadata for ALL the categories
    # NOTE: There should be NO Refreshes in the logs
    pprint(aws_broker.get_all_metadata())
