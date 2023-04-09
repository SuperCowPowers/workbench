"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod

import time
import logging
from typing import final

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.sageworks_logging import logging_setup

# Set up logging
logging_setup()


class Connector(ABC):
    # Class attributes
    log = logging.getLogger(__name__)

    # Set up our Boto3 and SageMaker Session and SageMaker Client
    boto_session = AWSAccountClamp().boto_session()
    sm_session = AWSAccountClamp().sagemaker_session()
    sm_client = AWSAccountClamp().sagemaker_client()

    # Set up the token refresh time
    refresh_minutes = 45
    token_refresh_time = time.time() + (refresh_minutes * 60)

    @classmethod
    def refresh_class_aws_sessions(cls):
        """Class method to refresh our AWS Sessions/Tokens"""
        now = time.time()
        if now > cls.token_refresh_time:
            cls.log.info(f"Current Refresh Time: {cls.token_refresh_time}")
            cls.token_refresh_time = now + (cls.refresh_minutes * 60)
            cls.log.info(f"New Refresh Time: {cls.token_refresh_time}")
            cls.log.info("Refreshing AWS SSO Token...")
            cls.boto_session = AWSAccountClamp().boto_session()
            cls.sm_session = AWSAccountClamp().sagemaker_session()
            cls.sm_client = AWSAccountClamp().sagemaker_client()

    def __init__(self):
        """Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
        self.log = logging.getLogger(__name__)

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
        Connector.refresh_class_aws_sessions()

        # Call the subclass Refresh method
        return self.refresh_impl()

    @abstractmethod
    def metadata(self) -> dict:
        """Return all the metadata for this service"""
        pass
