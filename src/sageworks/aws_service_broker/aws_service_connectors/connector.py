"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod

import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager
from sageworks.utils.sageworks_logging import logging_setup

# Set up logging
logging_setup()


class Connector(ABC):
    def __init__(self):
        """Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
        self.log = logging.getLogger(__name__)

        # Set up our Boto3 and SageMaker Session and SageMaker Client
        self.boto_session = AWSSageWorksRoleManager().boto_session()
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()
        self.sm_client = self.sm_session.boto_session.client("sagemaker")

        # FIXME: Figure out our SageWorks Artifacts Bucket Name

    @abstractmethod
    def check(self) -> bool:
        """Can we connect to this service?"""
        pass

    @abstractmethod
    def refresh(self) -> bool:
        """Refresh data/metadata associated with this service"""
        pass

    @abstractmethod
    def metadata(self) -> dict:
        """Return all the metadata for this service"""
        pass
