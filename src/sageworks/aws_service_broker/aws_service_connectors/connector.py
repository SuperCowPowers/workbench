"""Connector: Abstract Base Class for pulling/refreshing AWS Service metadata"""
from abc import ABC, abstractmethod
import botocore
import time

import logging
from typing import final

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp


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

    @staticmethod
    def _aws_tags_to_dict(aws_tags) -> dict:
        """Internal: AWS Tags are in an odd format, so convert to regular dictionary"""
        return {item["Key"]: item["Value"] for item in aws_tags}

    def sageworks_meta_via_arn(self, arn: str) -> dict:
        """Helper: Get the SageWorks specific metadata for this ARN
        Note: This functionality is a helper or Feature Store, Models, and Endpoints.
              The Data Catalog and Glue Jobs class have their own methods/logic
        """
        # Note: AWS List Tags can get grumpy if called too often
        self.log.debug(f"Calling list_tags AWS request {arn}...")
        try:
            aws_tags = self.sm_session.list_tags(arn)
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                self.log.warning(f"ThrottlingException: list_tags on {arn}")
                time.sleep(5)
                aws_tags = self.sm_session.list_tags(arn)
            else:
                # Handle other ClientErrors that may occur
                self.log.error(f"Caught a different ClientError: {error_code}")
        meta = self._aws_tags_to_dict(aws_tags)
        return meta
