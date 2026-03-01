"""PageView: Pulls from the Cloud Metadata and performs page specific data processing"""

from abc import ABC, abstractmethod

import logging
import pandas as pd

from workbench.utils.datetime_utils import concise_timestamps


class PageView(ABC):
    def __init__(self):
        """PageView: Pulls from the Cloud Metadata and performs page specific data processing"""
        self.log = logging.getLogger("workbench")

    @abstractmethod
    def refresh(self):
        """Refresh the data associated with this page view"""
        pass

    @staticmethod
    def concise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Format all datetime columns to concise minute-resolution strings (YYYY-MM-DD HH:MM).

        Args:
            df (pd.DataFrame): The DataFrame to format

        Returns:
            pd.DataFrame: The DataFrame with formatted datetime columns
        """
        return concise_timestamps(df)
