"""Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts.
The Meta class provides 'meta' information, what account are we in, what is the current
configuration, etc. It also provides summaries of the AWS Artifacts, such as Data Sources,
Feature Sets, Models, and Endpoints.
"""

import logging

import pandas as pd

# SageWorks Imports
from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker, ServiceCategory
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.datetime_utils import datetime_string
from sageworks.utils.aws_utils import num_columns_ds, aws_url


class Meta:
    """Meta: A class that provides high level information and summaries of SageWorks/AWS Artifacts

    Common Usage:
    ```
    meta = Meta()
    meta.account()
    meta.config()
    meta.data_sources()
    ```
    """

    def __init__(self):
        """Meta Initialization"""
        self.log = logging.getLogger("sageworks")

        # Account and Service Brokers
        self.aws_account = AWSAccountClamp()
        self.aws_broker = AWSServiceBroker()
        self.cm = ConfigManager()

    def account(self) -> dict:
        """Print out the AWS Account Info

        Returns:
            dict: The AWS Account Info
        """
        return self.aws_account.get_aws_account_info()

    def config(self) -> dict:
        """Return the current SageWorks Configuration

        Returns:
            dict: The current SageWorks Configuration
        """
        return self.cm.get_all_config()

    def data_sources(self) -> pd.DataFrame:
        """Get a summary of the Data Sources in AWS

        Returns:
            pd.DataFrame: A summary of the Data Sources in AWS
        """
        data = self.data_sources_as_dict()
        data_summary = []

        # Pull in various bits of metadata for each data source
        for name, info in data.items():
            size = "-"
            summary = {
                "Name": name,
                "Size(MB)": size,
                "Modified": datetime_string(info.get("UpdateTime")),
                "Num Columns": num_columns_ds(info),
                "DataLake": info.get("IsRegisteredWithLakeFormation", "-"),
                "Tags": info.get("Parameters", {}).get("sageworks_tags", "-"),
                "Input": str(
                    info.get("Parameters", {}).get("sageworks_input", "-"),
                ),
                "_aws_url": aws_url(info, "DataSource"),  # Hidden Column
            }
            data_summary.append(summary)

        # Return the summary
        return pd.DataFrame(data_summary)

    def data_sources_as_dict(self, refresh: bool = False, include_sageworks_meta: bool = False) -> dict:
        """Get a summary of the Data Sources in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.
            include_sageworks_meta (bool, optional): Include the verbose SageWorks Metadata. Defaults to False.

        Returns:
            dict: A summary of the Data Sources in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.DATA_CATALOG)
        data = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG)

        # Data Sources are in the 'sageworks' database
        data = data["sageworks"]

        # Include the verbose SageWorks Metadata?
        if include_sageworks_meta:
            return data

        # Remove the SageWorks Metadata
        data = self._remove_sageworks_meta(data)
        return data

    def feature_sets_as_dict(self, refresh: bool = False) -> dict:
        """Get a summary of the Feature Sets in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Feature Sets in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.FEATURE_STORE)
        return self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE)

    def models_as_dict(self, refresh: bool = False) -> dict:
        """Get a summary of the Models in AWS

         Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Models in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.MODELS)
        return self.aws_broker.get_metadata(ServiceCategory.MODELS)

    def endpoints_ad_dict(self, refresh: bool = False) -> dict:
        """Get a summary of the Endpoints in AWS

        Args:
            refresh (bool, optional): Force a refresh of the metadata. Defaults to False.

        Returns:
            dict: A summary of the Endpoints in AWS
        """
        if refresh:
            self.aws_broker.refresh_aws_data(ServiceCategory.ENDPOINTS)
        return self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS)

    def _remove_sageworks_meta(self, data: dict) -> dict:
        """Internal: Recursively remove any keys with 'sageworks_' in them"""

        # Recursively exclude any keys with 'sageworks_' in them
        summary_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                summary_data[key] = self._remove_sageworks_meta(value)
            elif not key.startswith("sageworks_"):
                summary_data[key] = value
        return summary_data


if __name__ == "__main__":
    """Exercise the SageWorks Meta Class"""
    from pprint import pprint

    # Create the class
    meta = Meta()

    # Get the AWS Account Info
    print("*** AWS Account ***")
    pprint(meta.account())

    # Get the SageWorks Configuration
    print("*** SageWorks Configuration ***")
    pprint(meta.config())

    # Get the Data Sources
    print("\n\n*** Data Sources ***")
    pprint(meta.data_sources())

    # Get the Feature Sets
    print("\n\n*** Feature Sets ***")
    pprint(meta.feature_sets())

    # Get the Models
    print("\n\n*** Models ***")
    pprint(meta.models())

    # Get the Endpoints
    print("\n\n*** Endpoints ***")
    pprint(meta.endpoints())
