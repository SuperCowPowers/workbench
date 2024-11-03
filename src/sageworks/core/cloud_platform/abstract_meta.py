"""AbstractMeta: An abstract class that provides high level information and summaries of Cloud Platform Artifacts.
The AbstractMeta class provides 'account' information, configuration, etc. It also provides metadata for Artifacts,
such as Data Sources, Feature Sets, Models, and Endpoints.
"""

import sys
import logging
from typing import Union
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Iterable

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.utils.config_manager import ConfigManager


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

    @abstractmethod
    def models(self) -> pd.DataFrame:
        """Get a summary of the Models deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Models deployed in the Cloud Platform
        """
        pass

    @abstractmethod
    def endpoints(self) -> pd.DataFrame:
        """Get a summary of the Endpoints deployed in the Cloud Platform

        Returns:
            pd.DataFrame: A summary of the Endpoints in the Cloud Platform
        """
        pass

    @abstractmethod
    def glue_job(self, job_name: str) -> Union[dict, None]:
        """Get the details of a specific Glue Job

        Args:
            job_name (str): The name of the Glue Job

        Returns:
            dict: The details of the Glue Job (None if not found)
        """
        pass

    @abstractmethod
    def data_source(self, table_name: str, database: str = "sageworks") -> Union[dict, None]:
        """Get the details of a specific Data Source

        Args:
            table_name (str): The name of the Data Source
            database (str, optional): The Glue database. Defaults to 'sageworks'.

        Returns:
            dict: The details of the Data Source (None if not found)
        """
        pass

    @abstractmethod
    def feature_set(self, feature_group_name: str) -> Union[dict, None]:
        """Get the details of a specific Feature Set

        Args:
            feature_group_name (str): The name of the Feature Set

        Returns:
            dict: The details of the Feature Set (None if not found)
        """
        pass

    @abstractmethod
    def model(self, model_name: str) -> Union[dict, None]:
        """Get the details of a specific Model

        Args:
            model_name (str): The name of the Model

        Returns:
            dict: The details of the Model (None if not found)
        """
        pass

    @abstractmethod
    def endpoint(self, endpoint_name: str) -> Union[dict, None]:
        """Get the details of a specific Endpoint

        Args:
            endpoint_name (str): The name of the Endpoint

        Returns:
            dict: The details of the Endpoint (None if not found)
        """
        pass

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
