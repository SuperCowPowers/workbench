"""WebView: A View in the database sense: Pulls from the AWS Service Broker and does slice and dice"""

from abc import ABC, abstractmethod

import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class WebView(ABC):
    def __init__(self):
        """WebView: A View in the database sense: Pulls from the AWS Service Broker and does slice and dice"""
        self.log = logging.getLogger("sageworks")

        # Grab an AWS Metadata Broker object for pulling AWS Service information
        self.aws_broker = AWSServiceBroker()
        self.aws_account_clamp = AWSAccountClamp()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.sm_session = self.aws_account_clamp.sagemaker_session()

    @abstractmethod
    def refresh(self) -> bool:
        """Refresh data/metadata associated with this view"""
        pass

    @abstractmethod
    def view_data(self) -> dict:
        """Return all the data that's useful for this view"""
        pass
