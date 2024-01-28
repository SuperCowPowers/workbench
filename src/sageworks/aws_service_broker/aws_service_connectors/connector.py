"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""

from abc import ABC, abstractmethod

import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


class Connector(ABC):
    """Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""

    def __init__(self):
        """Initialize the Connector Base Class"""

        # Initializing class attributes
        self.log = logging.getLogger("sageworks")
        self.aws_account_clamp = AWSAccountClamp()
        self.boto_session = self.aws_account_clamp.boto_session()
        self.sm_session = self.aws_account_clamp.sagemaker_session(self.boto_session)
        self.sm_client = self.aws_account_clamp.sagemaker_client(self.boto_session)

    @abstractmethod
    def check(self) -> bool:
        """Can we connect to this service?"""
        pass

    @abstractmethod
    def refresh(self):
        """Refresh logic for AWS Service Data"""
        pass

    @abstractmethod
    def summary(self) -> dict:
        """Return a summary list of all the AWS resources for this service"""
        pass

    @abstractmethod
    def details(self, name: str) -> dict:
        """Return the details for a specific AWS resource
        Args:
            name (str): The name of the AWS resource
        Returns:
            dict: A dictionary of details about this AWS resource
        """
        pass
