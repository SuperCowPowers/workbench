"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod

import logging
from typing import final

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.aws_utils import list_tags_with_throttle


class Connector(ABC):
    # Class attributes
    log = logging.getLogger("sageworks")

    # Set up our Boto3 and SageMaker Session and SageMaker Client
    aws_account_clamp = AWSAccountClamp()
    boto_session = aws_account_clamp.boto_session()
    sm_session = aws_account_clamp.sagemaker_session(boto_session)
    sm_client = aws_account_clamp.sagemaker_client(boto_session)

    def __init__(self):
        """Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
        self.log = logging.getLogger("sageworks")

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
    def aws_meta(self) -> dict:
        """Return ALL the AWS metadata for this AWS Service"""
        pass

    def sageworks_meta_via_arn(self, arn: str) -> dict:
        """Helper: Get the SageWorks specific metadata for this ARN
        Args:
            arn (str): The ARN of the SageMaker resource
        Returns:
            dict: A dictionary of SageWorks specific metadata
        """
        return list_tags_with_throttle(arn, self.sm_session)
