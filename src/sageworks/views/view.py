"""View: View in the database sense: Pulls from the AWS Service Broker and does slice and dice"""
from abc import ABC, abstractmethod

import logging

# Local Imports
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.utils.logging import logging_setup

# Set up logging
logging_setup()


class View(ABC):
    def __init__(self, database_scope='all'):
        """View: View in the database sense: Pulls from the AWS Service Broker and does slice and dice"""
        self.log = logging.getLogger(__name__)

        # Grab an AWS Metadata Broker object and pull information for Data Sources
        self.aws_broker = AWSServiceBroker(database_scope=database_scope)

    @abstractmethod
    def check(self) -> bool:
        """Can we connect to this view/service?"""
        pass

    @abstractmethod
    def refresh(self) -> bool:
        """Refresh data/metadata associated with this view"""
        pass

    @abstractmethod
    def view_data(self) -> dict:
        """Return all the data that's useful for this view"""
        pass
