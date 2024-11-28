"""PageView: Pulls from the Cloud Metadata and performs page specific data processing"""

from abc import ABC, abstractmethod

import logging

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


class PageView(ABC):
    def __init__(self):
        """PageView: Pulls from the Cloud Metadata and performs page specific data processing"""
        self.log = logging.getLogger("sageworks")

        # Grab our AWS Account Clamp
        self.aws_account_clamp = AWSAccountClamp()
        self.boto3_session = self.aws_account_clamp.boto3_session
        self.sm_session = self.aws_account_clamp.sagemaker_session()

    @abstractmethod
    def refresh(self):
        """Refresh the data associated with this page view"""
        pass
