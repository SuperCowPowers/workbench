"""PageView: Pulls from the Cloud Metadata and performs page specific data processing"""

from abc import ABC, abstractmethod

import logging

# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


class PageView(ABC):
    def __init__(self):
        """PageView: Pulls from the Cloud Metadata and performs page specific data processing"""
        self.log = logging.getLogger("sageworks")

    @abstractmethod
    def refresh(self):
        """Refresh the data associated with this page view"""
        pass
