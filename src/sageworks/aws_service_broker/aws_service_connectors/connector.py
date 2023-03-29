"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod

import time
import logging
from typing import final

# SageWorks Imports
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager
from sageworks.utils.sageworks_logging import logging_setup

# Set up logging
logging_setup()


class Connector(ABC):
    def __init__(self, token_refresh=45):
        """Connector: Abstract Base Class for pulling/refreshing AWS Service metadata
           Args:
               token_refresh(str): AWS Token Refresh Time in minutes"""
        self.log = logging.getLogger(__name__)

        # Set up our Boto3 and SageMaker Session and SageMaker Client
        self.boto_session = AWSSageWorksRoleManager().boto_session()
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()
        self.sm_client = self.sm_session.boto_session.client("sagemaker")

        # Set up the token refresh time
        self.refresh_minutes = token_refresh
        self.token_refresh_time = time.time() + (self.refresh_minutes * 60)

    @abstractmethod
    def check(self) -> bool:
        """Can we connect to this service?"""
        pass

    @abstractmethod
    def refresh_impl(self):
        """Abstract Method: Implement the AWS Service Data Refresh"""
        pass

    @final
    def refresh(self) -> bool:
        """Refresh data/metadata associated with this service"""
        # Check if it's time to Refresh our AWS/SSO Token
        now = time.time()
        if now > self.token_refresh_time:
            self.log.info('Refreshing AWS SSO Token...')
            self.boto_session = AWSSageWorksRoleManager().boto_session()
            self.sm_session = AWSSageWorksRoleManager().sagemaker_session()
            self.sm_client = self.sm_session.boto_session.client("sagemaker")
            self.token_refresh_time = now + (self.refresh_minutes * 60)

        # Call the subclass Refresh method
        return self.refresh_impl()

    @abstractmethod
    def metadata(self) -> dict:
        """Return all the metadata for this service"""
        pass
