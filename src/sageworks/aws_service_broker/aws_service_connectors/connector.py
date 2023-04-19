"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod

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
    aws_account_clamp = AWSAccountClamp()
    boto_session = aws_account_clamp.boto_session()
    sm_session = aws_account_clamp.sagemaker_session(boto_session)
    sm_client = aws_account_clamp.sagemaker_client(boto_session)

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
        # We could do something here to refresh the AWS Session or whatever

        # Call the subclass Refresh method
        return self.refresh_impl()

    @abstractmethod
    def metadata(self) -> dict:
        """Return all the metadata for this service"""
        pass
