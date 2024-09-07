"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""

from abc import ABC, abstractmethod
from collections import defaultdict
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
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.sm_session = self.aws_account_clamp.sagemaker_session()
        self.sm_client = self.aws_account_clamp.sagemaker_client()
        self.metadata_size_info = defaultdict(dict)

    @abstractmethod
    def check(self) -> bool:
        """Can we connect to this service?"""
        pass

    @abstractmethod
    def refresh(self):
        """Refresh logic for AWS Service Data"""
        pass

    @abstractmethod
    def summary(self, include_details: bool = False) -> dict:
        """Return a summary list of all the AWS resources for this service

        Args:
            include_details (bool, optional): Include the details for each resource (defaults to False)
        """
        pass

    def get_metadata_sizes(self) -> dict:
        """Return the size of the metadata for each AWS Service"""
        return dict(self.metadata_size_info)

    def report_metadata_sizes(self) -> None:
        """Report the size of the metadata for each AWS Service"""
        connector_name = self.__class__.__name__
        for key, size in self.metadata_size_info.items():
            self.log.info(f"{connector_name}: {key} ({size})")
