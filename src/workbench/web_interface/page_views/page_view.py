"""PageView: Pulls from the Cloud Metadata and performs page specific data processing"""

from abc import ABC, abstractmethod

import logging


class PageView(ABC):
    def __init__(self):
        """PageView: Pulls from the Cloud Metadata and performs page specific data processing"""
        self.log = logging.getLogger("workbench")

    @abstractmethod
    def refresh(self):
        """Refresh the data associated with this page view"""
        pass
