"""AbstractMeta: An abstract class that provides high level information and summaries of Cloud Platform Artifacts.
The AbstractMeta class provides 'account' information, configuration, etc. It also provides metadata for Artifacts,
such as Data Sources, Feature Sets, Models, and Endpoints.
"""

import sys
import logging
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Iterable

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager
from sageworks.api.pipeline_manager import PipelineManager


class AbstractMeta(ABC):
    """AbstractMeta: A class that provides Metadata for a broad set of Cloud Platform Artifacts"""

    def __init__(self, use_cache: bool = False):
        """AbstractMeta Initialization

        Args:
            use_cache (bool, optional): Use a cache for the metadata. Defaults to False.
        """
        self.log = logging.getLogger("sageworks")

        # We will be caching the metadata?
        self.use_cache = use_cache

        # Account and Configuration
        self.account_clamp = AWSAccountClamp()
        self.cm = ConfigManager()

        # Pipeline Manager
        self.pipeline_manager = PipelineManager()

        # Storing the size of various metadata for tracking
        self.metadata_sizes = defaultdict(dict)

    def account(self) -> dict:
        """Cloud Platform Account Info

        Returns:
            dict: Cloud Platform Account Info
        """
        return self.account_clamp.get_aws_account_info()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return self.cm.get_all_config()

    @abstractmethod
    def incoming_data(self) -> pd.DataFrame:
        """Get summary data about data in the incoming raw data

        Returns:
            pd.DataFrame: A summary of the incoming raw data
        """
        pass

    @abstractmethod
    def etl_jobs(self) -> pd.DataFrame:
        """Get summary data about Extract, Transform, Load (ETL) Jobs

        Returns:
            pd.DataFrame: A summary of the ETL Jobs deployed in the Cloud Platform
        """
        pass

    @abstractmethod
    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Data Sources deployed in the Cloud Platform
        """
        pass

    @abstractmethod
    def views(self, database: str = "sageworks") -> pd.DataFrame:
        """Get a summary of the all the Views, for the given database, in AWS

        Args:
            database (str, optional): Glue database. Defaults to 'sageworks'.

        Returns:
            pd.DataFrame: A summary of all the Views, for the given database, in AWS
        """
        pass

    @abstractmethod
    def feature_sets(self) -> pd.DataFrame:
        """Get a summary of the Feature Sets deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Feature Sets deployed in the Cloud Platform
        """
        pass

    def models(self) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        pass

    def endpoints(self) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        pass

    def pipelines(self, refresh: bool = False) -> pd.DataFrame:
        """Get a summary of the SageWorks Pipelines

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            pd.DataFrame: A summary of the SageWorks Pipelines
        """
        data = self.pipeline_manager.list_pipelines()

        # Return the pipelines summary as a DataFrame
        return pd.DataFrame(data)

    def compute_size(self, obj: object) -> int:
        """Recursively calculate the size of an object including its contents.

        Args:
            obj (object): The object whose size is to be computed.

        Returns:
            int: The total size of the object in bytes.
        """
        if isinstance(obj, Mapping):
            return sys.getsizeof(obj) + sum(self.compute_size(k) + self.compute_size(v) for k, v in obj.items())
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
            return sys.getsizeof(obj) + sum(self.compute_size(item) for item in obj)
        else:
            return sys.getsizeof(obj)
