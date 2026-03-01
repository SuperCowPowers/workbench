"""PageView: Pulls from the Cloud Metadata and performs page specific data processing"""

from abc import ABC, abstractmethod

import logging
import pandas as pd


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
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
            elif df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[col] = parsed.dt.strftime("%Y-%m-%d %H:%M").where(parsed.notna(), df[col])
                except Exception:
                    pass
        return df
